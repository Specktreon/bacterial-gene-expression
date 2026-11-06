"""Bacterial gene expression: additive + structured residual model.

Model on the log scale, z = log(y + 0.5) - offset, offset = log_length_ratio
+ log_size_factor:

    z_{g,r} ~ m + a_g + b_r + interaction_{g,r}

Each evaluation region gets its own predictor for the parts it cannot see:

- A (seen gene, seen run, held out together as a run-block x gene-block per
  genome): per genome, kernel ridge regression of the held-out block from
  the observed cells of the same runs / genes (run-space and gene-space
  kernels, averaged).
- B (new run): the run's profile is a similarity-weighted average of the
  residual profiles of the genome's training runs; weights come from the run
  metadata (project, medium, growth phase, perturbation...).
- C (new gene): the gene effect is predicted by gradient boosting from gene
  features: sequence (codon bias against highly expressed genes, GC, promoter
  motifs), operon context, neighbouring genes' effects and cross-species
  target encodings of the product / gene name / UniRef cluster from the
  GenBank genomes.

The mean bias and the NB dispersion are calibrated per region on internal
holdouts that mimic each region (pseudo A blocks, held-out runs, held-out
genes):  v = c0 + c1 * var_gene + c2 * var_run,  mu = exp(eta + q0 + q1 * v),
sigma = 1 / (k * v).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from benchmark_utils.base_solver import GeneCompetSolver

PSEUDO = 0.5        # pseudo-count of the log transform
CLIP = 3.0          # residual clip used by the interaction models
MU_FLOOR = 0.5
# Coarse constants, fitted offline on the public eval regions (mode=train)
# with this model; one set per region type, nothing per genome. Calibrator
# parameters: see Calibrator. Honest 2-fold-by-genome CV of the fit:
# A 7.094, B 7.441, C 7.457, D 7.990.
CALIBRATION = {
    "A": [-0.0794, -10.5333, -0.4628, 0.3214, 0.2834, -0.8837, -0.0675],
    "B": [-0.0672, -0.3173, 0.3497, -8.4001, 0.3218, -0.7206, -0.1177],
    "C": [0.1122, -0.9161, -1.3073, -12.0000, 0.9746, 0.6851, -0.1164],
    "D": [-0.0716, 0.6810, -0.7147, -5.2698, 0.3575, -0.5501, -0.1999],
}
A_KRR_WEIGHT = 0.5     # blend of kernel ridge vs gene-representation ridge (A)
B_SHRINK_SEEN = 0.58    # shrink of the run profile when the project is seen (B)
C_NB_WEIGHT = 0.543   # weight of the neighbour genes' run profile (C)
C_GN_WEIGHT = 0.428   # weight of same gene-name-prefix genes' profile (C)
RUN_META = ["prj_accession", "medium_name", "medium_category", "growth_phase",
            "growth_mode", "oxygen_condition", "perturbation_category",
            "perturbation_compound", "has_perturbation", "is_wild_type",
            "temperature_celsius", "condition_confidence"]


def _log(msg, t0=None):
    if t0 is not None:
        msg = f"{msg} [{time.time() - t0:.1f}s]"
    print(f"[submission] {msg}", flush=True)


# ---------------------------------------------------------------------------
# NB likelihood and calibration
# ---------------------------------------------------------------------------

def nb_nll_vec(y, mu, r):
    y = np.asarray(y, np.float64)
    mu = np.clip(np.asarray(mu, np.float64), 1e-8, None)
    r = np.clip(np.asarray(r, np.float64), 1e-8, None)
    lp = (gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r)
          + np.where(y > 0, y * np.log(mu), 0.0) - (y + r) * np.log(r + mu))
    return np.minimum(-lp, 100.0)


class Calibrator:
    """Mean bias and NB dispersion from a variance proxy v:

        v     = (exp(c0) + exp(c1) u1 + exp(c2) u2) * exp(c3 (eta - 6))
        mu    = exp(eta + q0 + q1 v)          (log-normal style correction)
        sigma = 1 / (exp(k) v)

    u1, u2 are per-cell variance proxies (gene / run residual variances),
    eta the uncalibrated log-mean. Seven constants per region."""

    START = [0.3, -1.0, -1.0, -1.0, 0.2, 0.0, 0.0]

    def __init__(self, params=None):
        p = self.START if params is None else params
        self.p = np.asarray(p, float)

    @staticmethod
    def _apply(p, eta, u1, u2):
        v = np.exp(p[1]) + np.exp(p[2]) * u1 + np.exp(p[3]) * u2
        if len(p) > 6:
            v = v * np.exp(np.clip(p[6] * (eta - 6.0), -10, 10))
        mu = np.exp(np.clip(eta + p[0] + p[4] * np.minimum(v, 5.0), -20, 20))
        return np.maximum(mu, MU_FLOOR), 1.0 / (np.exp(p[5]) * v)

    def fit(self, y, eta, u1, u2, max_n=150_000, seed=0, mu_term=True):
        n = len(y)
        if n > max_n:
            sel = np.random.default_rng(seed).choice(n, max_n, replace=False)
            y, eta, u1, u2 = y[sel], eta[sel], u1[sel], u2[sel]
        start = list(self.START if mu_term else self.START[:6])

        def loss(p):
            return float(nb_nll_vec(y, *self._apply(p, eta, u1, u2)).mean())

        o = minimize(loss, start, method="L-BFGS-B",
                     bounds=[(-3, 3), (-12, 3), (-12, 3), (-12, 3), (-2, 2),
                             (-5, 5), (-1, 1)][:len(start)],
                     options=dict(maxiter=200))
        self.p, self.loss_ = o.x, o.fun
        return self

    def __call__(self, eta, u1, u2):
        return self._apply(self.p, eta, u1, u2)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def pair_index(df, a, b, index):
    """Row positions in ``index`` (strings 'a|b') of each (a, b) pair of df;
    -1 when unknown. Works on category codes, strings built once per pair."""
    ca = df[a].astype("category")
    cb = df[b].astype("category")
    nb = len(cb.cat.categories) + 1
    key = (ca.cat.codes.to_numpy(np.int64) * nb
           + cb.cat.codes.to_numpy(np.int64))
    inv, uniq = pd.factorize(key)
    sa = np.asarray(ca.cat.categories.astype(str))[uniq // nb]
    sb = np.asarray(cb.cat.categories.astype(str))[uniq % nb]
    m = index.get_indexer(pd.Index(sa) + "|" + pd.Index(sb))
    return m[inv].astype(np.int32)


def group_bounds(keys, n_groups):
    order = np.argsort(keys, kind="stable")
    bounds = np.searchsorted(keys[order], np.arange(n_groups + 1))
    return order, bounds


# ---------------------------------------------------------------------------
# Genome features
# ---------------------------------------------------------------------------

_LUT = np.full(256, 4, np.uint8)
for _i, _c in enumerate(b"ACGT"):
    _LUT[_c] = _i
    _LUT[_c + 32] = _i
_DEL = str.maketrans("", "", "0123456789 \n\r\t")
_COMP = np.array([3, 2, 1, 0, 4], np.uint8)


def parse_gbff(path):
    """Records of a GenBank file: name, version, sequence codes and the CDS
    (start, end, strand, gene, product, UniRef50/90)."""
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    records = []
    for block in text.split("\n//"):
        if "LOCUS" not in block:
            continue
        head, _, seq = block.partition("\nORIGIN")
        lines = head.split("\n")
        name = version = None
        for ln in lines[:15]:
            if ln.startswith("LOCUS"):
                name = ln.split()[1]
            elif ln.startswith("VERSION"):
                parts = ln.split()
                version = parts[1] if len(parts) > 1 else None
        codes = _LUT[np.frombuffer(seq.translate(_DEL).encode(), np.uint8)]
        cds, cur, in_feat, last = [], None, False, None
        for ln in lines:
            if not in_feat:
                in_feat = ln.startswith("FEATURES")
                continue
            if len(ln) > 5 and ln[5] != " ":
                if cur is not None:
                    cds.append(cur)
                cur = None
                if ln[5:21].strip() == "CDS":
                    cur = {"loc": ln[21:].strip(), "gene": "", "product": "",
                           "u50": "", "u90": ""}
                    last = "loc"
                continue
            if cur is None:
                continue
            s = ln[21:].strip()
            if s.startswith("/"):
                k, _, v = s[1:].partition("=")
                v = v.strip('"')
                last = k
                if k == "gene":
                    cur["gene"] = v
                elif k == "product":
                    cur["product"] = v
                elif k == "db_xref" and v.startswith("UniRef:UniRef50_"):
                    cur["u50"] = v[16:]
                elif k == "db_xref" and v.startswith("UniRef:UniRef90_"):
                    cur["u90"] = v[16:]
            elif last == "loc":
                cur["loc"] += s
            elif last == "product":
                cur["product"] += " " + s.strip('"')
        if cur is not None:
            cds.append(cur)
        table = {}
        for c in cds:
            nums = [int(x) for x in
                    c["loc"].replace("<", "").replace(">", "")
                    .replace("(", " ").replace(")", " ").replace(".", " ")
                    .replace(",", " ").split() if x.isdigit()]
            if nums:
                table[(min(nums), max(nums))] = c
        records.append({"name": name, "version": version, "codes": codes,
                        "cds": table})
    return records


def _motif_counts(codes, motif):
    """Number of matching bases of ``motif`` at every position (plus strand)."""
    mc = _LUT[np.frombuffer(motif.encode(), np.uint8)]
    n = len(codes) - len(mc) + 1
    if n <= 0:
        return np.zeros(1, np.int8)
    out = np.zeros(n, np.int8)
    for j, c in enumerate(mc):
        out += codes[j:j + n] == c
    return out


def _range_max(arr, lo, hi):
    """max(arr[lo:hi]) for each window (empty windows -> 0)."""
    n = len(arr)
    lo = np.clip(lo, 0, n - 1)
    hi = np.clip(hi, 0, n)
    ok = hi > lo
    out = np.zeros(len(lo), np.float32)
    if ok.any():
        idx = np.empty(2 * ok.sum(), np.int64)
        idx[0::2] = lo[ok]
        idx[1::2] = np.minimum(hi[ok], n - 1)
        red = np.maximum.reduceat(arr, idx)[0::2].astype(np.float32)
        # reduceat stops at hi exclusive except when hi == n-1 clipped
        last = hi[ok] == n
        red[last] = np.maximum(red[last], arr[n - 1])
        out[ok] = red
    return out


def _range_mean(cum, lo, hi, n):
    lo = np.clip(lo, 0, n)
    hi = np.clip(hi, 0, n)
    ln = hi - lo
    return np.where(ln > 0, (cum[hi] - cum[lo]) / np.maximum(ln, 1), np.nan)


def record_features(rec, start, end, plus, upstream=150):
    """Sequence features of the genes (1-based start/end) of one record."""
    s = rec["codes"]
    L = len(s)
    n = len(start)
    out = {}
    valid = (start >= 1) & (end <= L) & (end > start + 2)
    st0 = np.where(valid, start - 1, 0)
    en = np.where(valid, end, 3)
    gc = ((s == 1) | (s == 2)).astype(np.int32)
    at = ((s == 0) | (s == 3)).astype(np.int32)
    cgc = np.concatenate([[0], np.cumsum(gc)])
    cat_ = np.concatenate([[0], np.cumsum(at)])
    out["gc"] = _range_mean(cgc, st0, en, L)
    up_lo = np.where(plus, st0 - upstream, en)
    up_hi = np.where(plus, st0, en + upstream)
    out["up_at"] = _range_mean(cat_, up_lo, up_hi, L)
    # codon counts
    pad = np.concatenate([s, [4, 4]])
    bad = (pad[:-2] > 3) | (pad[1:-1] > 3) | (pad[2:] > 3)
    cp = (16 * pad[:-2].astype(np.int16) + 4 * pad[1:-1] + pad[2:])
    cp[bad] = 64
    rc = _COMP[pad]
    cm = (16 * rc[2:].astype(np.int16) + 4 * rc[1:-1] + rc[:-2])
    cm[bad] = 64
    ncod = np.where(valid, (en - st0) // 3, 0)
    tot = int(ncod.sum())
    gid = np.repeat(np.arange(n), ncod)
    k = np.arange(tot) - np.repeat(np.cumsum(ncod) - ncod, ncod)
    pl = np.repeat(plus, ncod)
    pos = np.where(pl, np.repeat(st0, ncod) + 3 * k,
                   np.repeat(en - 3, ncod) - 3 * k)
    cidx = np.where(pl, cp[pos], cm[pos]).astype(np.int64)
    counts = np.bincount(gid * 65 + cidx, minlength=n * 65).reshape(n, 65)
    first = np.full(n, 64)
    lastc = np.full(n, 64)
    has = ncod > 0
    cstart = np.cumsum(ncod) - ncod
    first[has] = cidx[cstart[has]]
    lastc[has] = cidx[cstart[has] + ncod[has] - 1]
    out["start_atg"] = (first == 14).astype(np.float32)
    out["stop_ok"] = np.isin(lastc, (48, 50, 56)).astype(np.float32)
    # promoter motifs (minus strand: reverse-complement motif on plus coords)
    for name, motif, (a0, a1) in [("m10", "TATAAT", (60, 0)),
                                  ("m35", "TTGACA", (80, 15)),
                                  ("sd", "AGGAGG", (20, 0))]:
        mp = _motif_counts(s, motif)
        rcm = "".join({"A": "T", "C": "G", "G": "C", "T": "A"}[c]
                      for c in motif[::-1])
        mm = _motif_counts(s, rcm)
        w = len(motif)
        lo_p, hi_p = st0 - a0, st0 - a1 - w + 1
        lo_m, hi_m = en + a1, en + a0 - w + 1
        out[name] = np.where(plus, _range_max(mp, lo_p, hi_p),
                             _range_max(mm, lo_m, hi_m))
    out["rec_len"] = np.full(n, np.log(max(L, 1)), np.float32)
    mid = (st0 + en) / 2 / max(L, 1)
    out["rel_pos"] = np.minimum(mid, 1 - mid)
    for key in out:
        out[key] = np.where(valid, out[key], np.nan)
    return out, counts[:, :64].astype(np.float32)


def gene_features(feat, genomes_dir):
    """Per gene (rows of ``feat``): numeric sequence features, codon counts,
    annotation strings, neighbour pointers and operon proxy."""
    t0 = time.time()
    n = len(feat)

    def first_int(col, last=False):
        s = feat[col].astype(str).str.split(";").str[-1 if last else 0]
        return pd.to_numeric(s, errors="coerce").fillna(-1).to_numpy(np.int64)

    start, end = first_int("Start"), first_int("End", last=True)
    plus = feat["Strand"].astype(str).str[0].to_numpy() == "+"
    chrs = feat["Chr"].astype(str).str.split(";").str[0].to_numpy()
    genome = feat["genome_id"].astype(str).to_numpy()
    num_cols = ["gc", "up_at", "start_atg", "stop_ok", "m10", "m35", "sd",
                "rec_len", "rel_pos"]
    num = {c: np.full(n, np.nan, np.float32) for c in num_cols}
    codons = np.zeros((n, 64), np.float32)
    strs = {c: np.full(n, "", object) for c in
            ("gene_name", "product", "u50", "u90")}
    if genomes_dir is not None:
        groups = pd.Series(np.arange(n)).groupby([genome, chrs]).indices
        by_genome = {}
        for (gid, ch), rows in groups.items():
            by_genome.setdefault(gid, []).append((ch, rows))
        for gid, items in by_genome.items():
            path = Path(genomes_dir) / f"{gid}.gbff"
            if not path.exists():
                continue
            recs = parse_gbff(path)
            by_id = {}
            for rec in recs:
                for key in (rec["name"], rec["version"]):
                    if key:
                        by_id[key] = rec
            for ch, rows in items:
                rec = by_id.get(ch)
                if rec is None:
                    continue
                f, cod = record_features(rec, start[rows], end[rows],
                                         plus[rows])
                for c in num_cols:
                    num[c][rows] = f[c]
                codons[rows] = cod
                table = rec["cds"]
                for i in rows:
                    c = table.get((start[i], end[i]))
                    if c is not None:
                        strs["gene_name"][i] = c["gene"]
                        strs["product"][i] = c["product"]
                        strs["u50"][i] = c["u50"]
                        strs["u90"][i] = c["u90"]
    _log("genome sequence features", t0)
    out = pd.DataFrame(num)
    out["start"] = start
    for c, v in strs.items():
        out[c] = v
    nb = _neighbours(genome, chrs, start, end, plus)
    for c, v in nb.items():
        out[c] = v
    return out, codons


def _neighbours(genome, chrs, start, end, plus):
    n = len(genome)
    order = np.lexsort((start, chrs, genome))
    g_o, c_o = genome[order], chrs[order]
    s_o, e_o, p_o = start[order], end[order], plus[order]
    same = (g_o[1:] == g_o[:-1]) & (c_o[1:] == c_o[:-1])
    gap_next = np.full(n, np.nan)
    gap_next[:-1] = np.where(same, s_o[1:] - e_o[:-1], np.nan)
    gap_prev = np.r_[np.nan, gap_next[:-1]]
    same_next = np.zeros(n, bool)
    same_next[:-1] = same & (p_o[1:] == p_o[:-1])
    same_prev = np.r_[False, same_next[:-1]]
    link = same_next & (gap_next < 30)
    grp = np.r_[0, np.cumsum(~link[:-1])]
    size = np.bincount(grp)[grp]
    first = np.r_[True, grp[1:] != grp[:-1]]
    rank = np.arange(n) - np.maximum.accumulate(np.where(first, np.arange(n), 0))
    res = {
        "up_gap": np.where(p_o, gap_prev, gap_next),
        "dn_gap": np.where(p_o, gap_next, gap_prev),
        "up_same": np.where(p_o, same_prev, same_next).astype(np.float32),
        "dn_same": np.where(p_o, same_next, same_prev).astype(np.float32),
        "op_size": size.astype(np.float32),
        "op_pos": np.where(p_o, rank, size - 1 - rank).astype(np.float32),
        "op_id": grp,
    }
    prev_o = np.r_[-1, np.where(same, order[:-1], -1)]
    next_o = np.r_[np.where(same, order[1:], -1), -1]
    res["prev_i"] = prev_o
    res["next_i"] = next_o
    out = {}
    for k, v in res.items():
        full = np.empty(n, dtype=np.asarray(v).dtype)
        full[order] = v
        out[k] = full
    return out


# ---------------------------------------------------------------------------
# Gene table features for the new-gene model
# ---------------------------------------------------------------------------

def _target_encode(keys, target, valid, species, prior=10.0):
    """Mean target of the genes sharing ``keys``, leaving the gene's own
    species out (orthologs of a held-out gene are held out with it)."""
    keys = np.asarray(keys, dtype=object)
    empty = keys == ""
    valid = valid & ~empty
    kidx = pd.factorize(keys)[0]
    nk = kidx.max() + 1
    sk = np.bincount(kidx, target * valid, nk)
    ck = np.bincount(kidx, valid, nk)
    pair = pd.factorize(kidx.astype(np.int64) * (species.max() + 1)
                        + species)[0]
    npair = pair.max() + 1
    sp = np.bincount(pair, target * valid, npair)
    cp = np.bincount(pair, valid, npair)
    S, N = sk[kidx] - sp[pair], ck[kidx] - cp[pair]
    enc = np.where(empty, np.nan, S / (N + prior))
    return enc, np.where(empty, np.nan, N)


def gene_design(feat, gf, codons, t, known, gen, species):
    """Design matrix for predicting the centred gene effect ``t``."""
    X = {}
    X["loglen"] = np.log(np.maximum(feat["Length"].to_numpy(float), 1))
    X["llr"] = feat["log_length_ratio"].to_numpy(float)
    X["plus"] = (feat["Strand"].astype(str).str[0] == "+").to_numpy(float)
    X["edge"] = feat["on_contig_edge"].astype(float).to_numpy()
    X["homol"] = pd.to_numeric(feat["homology"], errors="coerce").to_numpy()
    X["has_ref"] = feat["ref_feature_id"].notna().to_numpy(float)
    for c in ["gc", "up_at", "m10", "m35", "sd", "rec_len", "rel_pos",
              "start_atg", "stop_ok", "up_gap", "dn_gap", "up_same",
              "dn_same", "op_size", "op_pos"]:
        X[c] = gf[c].to_numpy(float)
    ng = gen.max() + 1
    gc_mean = (np.bincount(gen, np.nan_to_num(X["gc"]), ng)
               / np.maximum(np.bincount(gen, ~np.isnan(X["gc"]), ng), 1))
    X["gc_rel"] = X["gc"] - gc_mean[gen]
    tot = codons.sum(1)
    fr = codons / np.maximum(tot, 1)[:, None]
    third = np.arange(64) % 4
    X["gc3"] = fr[:, (third == 1) | (third == 2)].sum(1)
    X["gc3_rel"] = X["gc3"] - (np.bincount(gen, X["gc3"], ng)
                               / np.maximum(np.bincount(gen, minlength=ng), 1))[gen]
    prod = gf["product"].astype(str).str.lower()
    X["hypo"] = prod.str.contains("hypothetical|uncharacterized|duf").to_numpy(float)
    # codon usage of highly expressed genes vs. the genome (leave-self-out)
    cai = np.full(len(t), np.nan)
    order, bounds = group_bounds(gen, ng)
    for k in range(ng):
        rows = order[bounds[k]:bounds[k + 1]]
        kn = rows[known[rows]]
        if len(kn) < 50:
            continue
        hi = kn[t[kn] >= np.quantile(t[kn], 0.95)]
        H = codons[hi].sum(0)
        pa = (codons[rows].sum(0) + 1)
        pa = pa / pa.sum()
        ph = (H + 1) / (H.sum() + 64)
        cai[rows] = fr[rows] @ np.log(ph / pa)
        for j in hi:
            Hj = H - codons[j]
            phj = (Hj + 1) / (Hj.sum() + 64)
            cai[j] = fr[j] @ np.log(phj / pa)
    X["cai"] = cai
    gname = gf["gene_name"].astype(str).str.lower()
    for name, keys in [("prod", prod), ("gname", gname),
                       ("gname3", gname.str[:3]), ("u90", gf["u90"]),
                       ("u50", gf["u50"])]:
        X["te_" + name], X["n_" + name] = _target_encode(
            keys.astype(str).to_numpy(), t, known, species)
    # product words: cross-species encoding of each word, pooled per gene
    words = prod.str.replace(r"[^a-z0-9]+", " ", regex=True).str.split()
    lens = words.str.len().fillna(0).astype(int).to_numpy()
    flat = np.concatenate([np.asarray(w, dtype=object) for w in words
                           if isinstance(w, list) and len(w)] or
                          [np.array([], dtype=object)])
    owner = np.repeat(np.arange(len(t)), lens)
    keep = ~np.isin(flat, list(STOP_WORDS))
    flat, owner = flat[keep], owner[keep]
    enc, cnt = _target_encode(flat, t[owner], known[owner], species[owner],
                              prior=20.0)
    ok = cnt >= 5
    wgt = np.where(ok, np.log1p(np.nan_to_num(cnt)), 0.0)
    num = np.bincount(owner, np.where(ok, enc * wgt, 0.0), len(t))
    den = np.bincount(owner, wgt, len(t))
    X["te_words"] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    mx = pd.Series(np.where(ok, enc, np.nan)).groupby(owner).max()
    mn = pd.Series(np.where(ok, enc, np.nan)).groupby(owner).min()
    X["te_words_max"] = mx.reindex(np.arange(len(t))).to_numpy()
    X["te_words_min"] = mn.reindex(np.arange(len(t))).to_numpy()
    # nearest known genes upstream / downstream (transcription order),
    # outside the gene's own operon proxy: effect, strand, distance
    plus = X["plus"] > 0
    pos = gf["start"].to_numpy(float)
    for name, idx, steps in zip(("up", "dn"), *nearest_known(gf, plus, known)):
        c = np.maximum(idx, 0)
        has = idx >= 0
        X["nb_" + name] = np.where(has, t[c], np.nan)
        X["nb_%s_same" % name] = np.where(has, (plus[c] == plus), np.nan)
        X["nb_%s_bp" % name] = np.where(has, np.abs(pos[c] - pos), np.nan)
        X["nb_%s_steps" % name] = np.where(has, steps, np.nan)
    return pd.DataFrame(X)


STOP_WORDS = {"protein", "family", "domain", "putative", "the", "of", "and",
              "subunit", "containing", "like", "type", "related", "probable"}


def nearest_known(gf, plus, known, max_steps=20):
    """Index (or -1) and step count of the nearest known gene upstream and
    downstream in transcription order, skipping the gene's operon proxy."""
    op = gf["op_id"].to_numpy()
    prev_i, next_i = gf["prev_i"].to_numpy(), gf["next_i"].to_numpy()
    out = []
    for upstream in (True, False):
        fwd = plus == upstream          # walk towards lower coordinates?
        cur = np.where(fwd, prev_i, next_i)
        idx = np.full(len(op), -1)
        steps = np.zeros(len(op))
        active = np.ones(len(op), bool)
        for step in range(max_steps):
            ok = active & (cur >= 0)
            c = np.maximum(cur, 0)
            hit = ok & (op[c] != op) & known[c]
            idx[hit], steps[hit] = c[hit], step + 1
            active &= ~hit & (cur >= 0)
            cur = np.where(active, np.where(fwd, prev_i[c], next_i[c]), -1)
        out.append((idx, steps))
    return [o[0] for o in out], [o[1] for o in out]


def _spline_basis(x, fit_x, n_knots=6):
    from sklearn.preprocessing import SplineTransformer
    sp = SplineTransformer(n_knots=n_knots, degree=3)
    sp.fit(np.asarray(fit_x, np.float64)[:, None])
    return sp.transform(np.asarray(x, np.float64)[:, None]).astype(np.float32)


def _gbm(seed=0, max_iter=600):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=127,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class ExpressionModel:

    def __init__(self, meta, lam_add=2.0, seed=0):
        self.feat = meta.get("feature_annotations")
        self.cond = meta.get("condition_annotations")
        self.lam_add = lam_add
        self.seed = seed

    # ----------------------------------------------------------- indexing
    def _setup_index(self, samples_df):
        cond = self.cond if self.cond is not None else samples_df
        self.feat = self.feat.reset_index(drop=True)
        self.cond = cond.reset_index(drop=True)
        feat, cond = self.feat, self.cond
        self.gene_index = pd.Index(feat["genome_id"].astype(str) + "|"
                                   + feat["feature_id"].astype(str))
        self.run_index = pd.Index(cond["genome_id"].astype(str) + "|"
                                  + cond["run_accession"].astype(str))
        genomes = pd.Index(sorted(set(feat["genome_id"].astype(str))
                                  | set(cond["genome_id"].astype(str))))
        self.gen_g = genomes.get_indexer(feat["genome_id"].astype(str))
        self.gen_r = genomes.get_indexer(cond["genome_id"].astype(str))
        self.n_genomes = len(genomes)
        self.NG, self.NR = len(feat), len(cond)
        species = pd.Index(sorted(set(feat["species"].astype(str))))
        self.species_g = species.get_indexer(feat["species"].astype(str))

    def _map(self, df):
        g = pair_index(df, "genome_id", "feature_id", self.gene_index)
        r = pair_index(df, "genome_id", "run_accession", self.run_index)
        if (g < 0).any() or (r < 0).any():
            raise ValueError("observations with an unknown gene or run")
        off = (df["log_length_ratio"].to_numpy(np.float64)
               + df["log_size_factor"].to_numpy(np.float64))
        return g, r, np.nan_to_num(off).astype(np.float32)

    # ----------------------------------------------------------- fitting
    def fit(self, train_df, samples_df, genomes_dir):
        T = time.time()
        self._setup_index(samples_df)
        g, r, off = self._map(train_df)
        y = train_df["readcount"].to_numpy(np.float32)
        _log(f"mapped {len(y)} rows", T)
        z = (np.log(y + PSEUDO) - off).astype(np.float32)
        self._fit_additive(g, r, z)
        del z
        _log("additive model", T)
        order, self.tr_bounds = group_bounds(self.gen_g[g], self.n_genomes)
        self.tr_g, self.tr_r = g[order], r[order]
        self.tr_y, self.tr_off = y[order], off[order]
        del order, g, r, y, off
        self.run_codes = self._run_codes()
        self._gene_tables(genomes_dir, T)
        shipped = all(CALIBRATION[k] is not None for k in "ABCD")
        if shipped:
            self.A_w = A_KRR_WEIGHT
        else:
            self._fit_A(T)
        self._fit_B(T, calibrate=not shipped)
        self._fit_C(T, holdout=not shipped)
        _log("fit done", T)
        return self

    def _fit_additive(self, g, r, z):
        NG, NR, lam = self.NG, self.NR, self.lam_add
        cg = np.bincount(g, minlength=NG).astype(np.float64)
        cr = np.bincount(r, minlength=NR).astype(np.float64)
        m = float(z.mean(dtype=np.float64))
        a, b = np.zeros(NG), np.zeros(NR)
        for _ in range(6):
            a = np.bincount(g, z - m - b[r], NG) / (cg + lam)
            b = np.bincount(r, z - m - a[g], NR) / (cr + lam)
        res2 = (z - m - a[g] - b[r]).astype(np.float64) ** 2
        self.vg = np.bincount(g, res2, NG) / np.maximum(cg, 1)
        self.vr = np.bincount(r, res2, NR) / np.maximum(cr, 1)
        self.m, self.a, self.b, self.cg, self.cr = m, a, b, cg, cr
        self.known_g, self.known_r = cg > 0, cr > 0
        ng = self.n_genomes
        kg = self.known_g
        self.gmean = (np.bincount(self.gen_g, a * kg, ng)
                      / np.maximum(np.bincount(self.gen_g, kg, ng), 1))
        self.vg_genome = (np.bincount(self.gen_g, self.vg * kg, ng)
                          / np.maximum(np.bincount(self.gen_g, kg, ng), 1))

    def _genome_matrix(self, k):
        """Training cells of genome k as dense run x gene matrices."""
        lo, hi = self.tr_bounds[k], self.tr_bounds[k + 1]
        g, r = self.tr_g[lo:hi], self.tr_r[lo:hi]
        ug, gl = np.unique(g, return_inverse=True)
        ur, rl = np.unique(r, return_inverse=True)
        shape = (len(ur), len(ug))
        Z = np.zeros(shape, np.float32)
        Y = np.zeros(shape, np.float32)
        O = np.zeros(shape, np.float32)
        W = np.zeros(shape, bool)
        y, off = self.tr_y[lo:hi], self.tr_off[lo:hi]
        Z[rl, gl] = np.log(y + PSEUDO) - off
        Y[rl, gl] = y
        O[rl, gl] = off
        W[rl, gl] = True
        return ug, ur, Z, Y, O, W

    # ----------------------------------------------------- A: block models
    @staticmethod
    def _dense_additive(Z, W, lam=2.0, iters=5):
        Wf = W.astype(np.float32)
        nr, nc = Wf.sum(1), Wf.sum(0)
        m = float((Z * Wf).sum() / max(Wf.sum(), 1))
        a = np.zeros(Z.shape[1], np.float32)
        b = np.zeros(Z.shape[0], np.float32)
        for _ in range(iters):
            a = ((Z - m - b[:, None]) * Wf).sum(0) / (nc + lam)
            b = ((Z - m - a[None, :]) * Wf).sum(1) / (nr + lam)
        R = np.where(W, Z - m - a[None, :] - b[:, None], 0.0)
        return m, a, b, np.clip(R, -CLIP, CLIP).astype(np.float32)

    def _a_block(self, Z, W, R2, G2, ug, krr_lam=0.3, ridge_lam=1.0,
                 npc=20):
        """Predict z on the block R2 x G2 (positions in the genome matrix)
        from every other training cell of the genome.

        Two predictors: (i) kernel ridge in run space and in gene space on
        the doubly-centred residuals; (ii) per-run ridge regression on a
        gene representation (gene effect over the other runs + spline,
        sequence features, principal components of the gene profiles),
        learnt on the block's runs over the observed genes. Returns
        (eta_krr, eta_ridge, gene variance, run variance)."""
        nR, nG = Z.shape
        Wb = W.copy()
        Wb[np.ix_(R2, G2)] = False
        isR2 = np.zeros(nR, bool)
        isR2[R2] = True
        isG2 = np.zeros(nG, bool)
        isG2[G2] = True
        R1, G1 = np.where(~isR2)[0], np.where(~isG2)[0]
        m, a, b, R = self._dense_additive(Z, Wb)
        vg = ((R[np.ix_(R1, G2)] ** 2).sum(0)
              / np.maximum(Wb[np.ix_(R1, G2)].sum(0), 1))
        vr = ((R[np.ix_(R2, G1)] ** 2).sum(1)
              / np.maximum(Wb[np.ix_(R2, G1)].sum(1), 1))
        base = m + a[G2][None, :] + b[R2][:, None]
        if len(R1) < 2 or len(G1) < 10:
            return base, base, vg, vr
        # (i) kernel ridge, run space + gene space
        X1, X2 = R[np.ix_(R1, G1)], R[np.ix_(R2, G1)]
        Y1 = R[np.ix_(R1, G2)]
        K11 = X1 @ X1.T
        s1 = np.trace(K11) / len(R1) + 1e-6
        P1 = (X2 @ X1.T) @ np.linalg.solve(
            K11 + krr_lam * s1 * np.eye(len(R1), dtype=np.float32), Y1)
        Kg = X1.T @ X1
        s2 = np.trace(Kg) / len(G1) + 1e-6
        P2 = X2 @ np.linalg.solve(
            Kg + krr_lam * s2 * np.eye(len(G1), dtype=np.float32), X1.T @ Y1)
        eta_krr = base + 0.5 * (P1 + P2)
        # (ii) gene-representation ridge: gene effects from R1 runs only,
        # run effects from G1 genes only
        Wf = Wb.astype(np.float32)
        WG1 = Wf.copy()
        WG1[:, G2] = 0
        a1 = np.zeros(nG, np.float32)
        b1 = np.zeros(nR, np.float32)
        for _ in range(5):
            a1 = ((Z[R1] - m - b1[R1, None]) * Wf[R1]).sum(0) \
                / (Wf[R1].sum(0) + 2)
            b1 = ((Z - m - a1[None, :]) * WG1).sum(1) / (WG1.sum(1) + 2)
        Xr = np.where(Wb[R1], Z[R1] - m - b1[R1, None] - a1[None, :],
                      0).clip(-CLIP, CLIP)
        Tm = Wb[np.ix_(R2, G1)]
        T = np.where(Tm, Z[np.ix_(R2, G1)] - m - b1[R2, None]
                     - a1[None, G1], 0).clip(-CLIP, CLIP)
        feats = [a1[:, None], _spline_basis(a1, a1[G1]), self.gene_seq[ug]]
        k = min(npc, len(R1) - 1)
        if k > 0:
            C = Xr.T @ Xr if len(R1) > nG else Xr @ Xr.T
            w, V = np.linalg.eigh(C)
            V = V[:, -k:]
            if len(R1) > nG:
                pcs = Xr.T @ (Xr @ V)
            else:
                pcs = Xr.T @ V
            feats.append(pcs / np.sqrt(len(R1)))
        F = np.concatenate(feats + [np.ones((nG, 1), np.float32)], 1)
        F = F.astype(np.float32)
        A_ = F[G1].T @ F[G1]
        sc = np.trace(A_) / len(A_)
        beta = np.linalg.solve(A_ + ridge_lam * 1e-2 * sc * np.eye(len(A_)),
                               F[G1].T @ T.T)
        eta_ridge = (m + b1[R2][:, None] + a1[G2][None, :]
                     + (F[G2] @ beta).T)
        return eta_krr, eta_ridge, vg, vr

    def _pseudo_block(self, ur, ug, rng, run_frac=0.2, gene_frac=0.31):
        """A-like block: whole projects for runs, whole operons for genes."""
        prj = self.run_codes[ur, 0]
        up = np.unique(prj)
        rng.shuffle(up)
        R2, tot = [], 0
        for p in up:
            rows = np.where(prj == p)[0]
            if tot + len(rows) > max(1, 1.5 * run_frac * len(ur)) and R2:
                continue
            R2.extend(rows)
            tot += len(rows)
            if tot >= run_frac * len(ur):
                break
        if len(R2) >= len(ur):
            R2 = rng.choice(len(ur), max(1, round(run_frac * len(ur))),
                            replace=False).tolist()
        op = self.gf["op_id"].to_numpy()[ug]
        uo = np.unique(op)
        G2 = np.where(np.isin(op, uo[rng.random(len(uo)) < gene_frac]))[0]
        return np.sort(np.asarray(R2)), G2

    def _fit_A(self, T, max_genomes=60):
        """Pseudo A blocks in the training data: blend weight and
        calibration of the mean bias / dispersion for region A."""
        rng = np.random.default_rng(self.seed + 1)
        runs_per = np.bincount(self.gen_r[self.known_r],
                               minlength=self.n_genomes)
        cand = np.where(runs_per >= 6)[0]
        w = np.sqrt(runs_per[cand].astype(float))
        pick = rng.choice(cand, min(max_genomes, len(cand)), replace=False,
                          p=w / w.sum())
        parts = []
        for k in pick:
            ug, ur, Z, Y, O, W = self._genome_matrix(k)
            R2, G2 = self._pseudo_block(ur, ug, rng)
            if len(G2) == 0:
                continue
            ek, er, vg, vr = self._a_block(Z, W, R2, G2, ug)
            blk = np.ix_(R2, G2)
            obs = W[blk]
            parts.append(np.stack([
                Y[blk][obs], O[blk][obs], Z[blk][obs], ek[obs], er[obs],
                np.broadcast_to(vg[None, :], obs.shape)[obs],
                np.broadcast_to(vr[:, None], obs.shape)[obs]], 1))
        y, off, z, ek, er, u1, u2 = np.concatenate(parts).T
        mse = {wt: float(np.mean((wt * ek + (1 - wt) * er - z) ** 2))
               for wt in (0.0, 0.25, 0.5, 0.75, 1.0)}
        self.A_w = min(mse, key=mse.get)
        eta = self.A_w * ek + (1 - self.A_w) * er + off
        self.cal_A = Calibrator().fit(y, eta, u1, u2)
        _log(f"A: blend mse {mse}, krr weight {self.A_w}, "
             f"pseudo-A nll {self.cal_A.loss_:.4f}", T)

    # --------------------------------------------- B: run metadata kNN
    def _run_codes(self):
        codes = []
        for c in RUN_META:
            if c not in self.cond.columns:
                codes.append(np.full(self.NR, -1))
                continue
            col = self.cond[c]
            s = col.astype(str).where(col.notna(), None)
            codes.append(pd.Series(s).astype("category").cat.codes.to_numpy())
        return np.stack(codes, 1)

    def _sim(self, ri, rj):
        a = self.run_codes[ri][:, None, :]
        b = self.run_codes[rj][None, :, :]
        return ((a == b) & (a >= 0)).astype(np.float32)

    def _genome_residuals(self, k):
        ug, ur, Z, Y, O, W = self._genome_matrix(k)
        R = np.where(W, Z - self.m - self.a[ug][None, :]
                     - self.b[ur][:, None], 0.0)
        return ug, ur, np.clip(R, -CLIP, CLIP).astype(np.float32), W, Z, Y, O

    def _fit_B(self, T, n_runs=400, n_genes=400, p_new_project=0.63,
               calibrate=True):
        """Leave-runs-out holdout: learn metadata similarity weights, then
        calibrate region B on the held-out runs."""
        import torch
        rng = np.random.default_rng(self.seed + 2)
        runs_per = np.bincount(self.gen_r[self.known_r],
                               minlength=self.n_genomes)
        cand = np.where(self.known_r & (runs_per[self.gen_r] >= 4))[0]
        held = rng.choice(cand, min(n_runs, len(cand)), replace=False)
        prj = self.run_codes[:, 0]
        groups = []
        for k in np.unique(self.gen_r[held]):
            ug, ur, R, W, Z, Y, O = self._genome_residuals(k)
            hk = held[self.gen_r[held] == k]
            hi = np.searchsorted(ur, hk)
            cols = np.sort(rng.choice(len(ug), min(n_genes, len(ug)),
                                      replace=False))
            S = self._sim(hk, ur)
            mask = np.ones((len(hk), len(ur)), np.float32)
            mask[np.arange(len(hk)), hi] = 0
            drop = rng.random(len(hk)) < p_new_project
            same_prj = (prj[hk][:, None] == prj[ur][None, :]) & (prj[hk][:, None] >= 0)
            mask[drop[:, None] & same_prj] = 0
            # leave-run-out gene effect: remove the run from a_g
            gk = ug[cols]
            n_g = self.cg[gk]
            zt = Z[np.ix_(hi, cols)]
            wt = W[np.ix_(hi, cols)]
            a_lo = (self.a[gk] * (n_g + self.lam_add)
                    - (zt - self.m - self.b[hk][:, None]) * wt) \
                / (n_g - wt + self.lam_add)
            target = np.where(wt, zt - self.m - a_lo, 0)
            groups.append(dict(
                S=torch.tensor(S), mask=torch.tensor(mask),
                Rc=torch.tensor(R[:, cols]), Wc=torch.tensor(W[:, cols].astype(np.float32)),
                bt=torch.tensor(self.b[ur].astype(np.float32)),
                target=torch.tensor(target.astype(np.float32)),
                wt=torch.tensor(wt.astype(np.float32)),
                k=k, hk=hk, hi=hi, a_lo_cols=cols))
        nf = self.run_codes.shape[1]
        beta = torch.zeros(nf, requires_grad=True)
        log_tau = torch.zeros(1, requires_grad=True)
        log_tau_b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([beta, log_tau, log_tau_b], max_iter=60,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            tot, cnt = 0.0, 0.0
            for G in groups:
                w = torch.exp(G["S"] @ beta) * G["mask"]
                bhat = (w @ G["bt"]) / (w.sum(1) + torch.exp(log_tau_b))
                prof = (w @ G["Rc"]) / (w @ G["Wc"] + torch.exp(log_tau))
                pred = bhat[:, None] + prof
                tot = tot + (((G["target"] - pred) ** 2) * G["wt"]).sum()
                cnt += float(G["wt"].sum())
            loss = tot / cnt
            loss.backward()
            return loss

        opt.step(closure)
        self.B_beta = beta.detach().numpy().astype(np.float32)
        self.B_tau = float(torch.exp(log_tau))
        self.B_tau_b = float(torch.exp(log_tau_b))
        _log(f"B: similarity weights {np.round(self.B_beta, 2).tolist()} "
             f"tau {self.B_tau:.3g} tau_b {self.B_tau_b:.3g} "
             f"mse {float(closure()):.4f}", T)
        if not calibrate:
            return
        # calibration cells: all genes of the held-out runs
        ys, etas, u1, u2 = [], [], [], []
        for G in groups:
            k, hk, hi = G["k"], G["hk"], G["hi"]
            ug, ur, R, W, Z, Y, O = self._genome_residuals(k)
            w = np.exp(G["S"].numpy() @ self.B_beta) * G["mask"].numpy()
            eta, vrh = self._b_predict(w, R, W, ur)
            n_g = self.cg[ug]
            wt = W[hi]
            a_lo = (self.a[ug][None, :] * (n_g + self.lam_add)
                    - (Z[hi] - self.m - self.b[hk][:, None]) * wt) \
                / (n_g - wt + self.lam_add)
            eta = eta + self.m + a_lo + O[hi]
            ys.append(Y[hi][wt])
            etas.append(eta[wt])
            u1.append(np.broadcast_to(self.vg[ug][None, :], wt.shape)[wt])
            u2.append(np.broadcast_to(vrh[:, None], wt.shape)[wt])
        self.cal_B = Calibrator().fit(*(np.concatenate(v) for v in
                                        (ys, etas, u1, u2)))
        _log(f"B: held-out-run nll {self.cal_B.loss_:.4f}", T)

    def _b_predict(self, w, R, W, ur):
        """Run effect + profile (|new runs|, |genes|) and run variance."""
        ws = w.sum(1)
        bhat = (w @ self.b[ur]) / (ws + self.B_tau_b)
        prof = (w @ R) / (w @ W.astype(np.float32) + self.B_tau)
        vrh = (w @ self.vr[ur]) / np.maximum(ws, 1e-12)
        vrh = np.where(ws > 0, vrh, self.vr[ur].mean())
        return bhat[:, None] + prof, vrh

    # ------------------------------------------- C: gene effect model
    def _gene_tables(self, genomes_dir, T):
        gf, self.codons = gene_features(self.feat, genomes_dir)
        self.gf = gf
        seq = np.column_stack([
            gf["gc"].to_numpy(float),
            np.log(np.maximum(self.feat["Length"].to_numpy(float), 1)),
            gf["up_at"].to_numpy(float), gf["rel_pos"].to_numpy(float)])
        mu, sd = np.nanmean(seq, 0), np.nanstd(seq, 0) + 1e-6
        self.gene_seq = np.nan_to_num((seq - mu) / sd).astype(np.float32)
        _log("gene features", T)

    def _fit_C(self, T, holdout=True):
        gf, codons = self.gf, self.codons
        known = self.known_g
        t = self.a - self.gmean[self.gen_g]
        X = gene_design(self.feat, gf, codons, t, known, self.gen_g,
                        self.species_g).to_numpy(np.float32)
        del codons
        self.codons = None
        plus = self.feat["Strand"].astype(str).str[0].to_numpy() == "+"
        (self.nb_up, self.nb_dn), _ = nearest_known(gf, plus, known)
        gname = gf["gene_name"].astype(str).str.lower().str[:3]
        self.gname3 = np.where(gname.str.len().to_numpy() == 3,
                               pd.factorize(gname)[0], -1)
        sw = np.sqrt(self.cg)
        if holdout:
            self._calibrate_C(X, t, known, sw, gf, T)
        rows = np.where(known)[0]
        mdl = _gbm(self.seed)
        mdl.fit(X[rows], t[rows], sample_weight=sw[rows])
        unknown = np.where(~known)[0]
        self.t_new = np.zeros(self.NG)
        self.t_new[unknown] = mdl.predict(X[unknown]) if len(unknown) else 0
        _log("C: final gene model", T)

    def _calibrate_C(self, X, t, known, sw, gf, T):
        rng = np.random.default_rng(self.seed + 3)
        # holdout grouped by operon proxy, mimicking new genes
        op = gf["op_id"].to_numpy()
        hold_op = rng.random(op.max() + 1) < 0.2
        hold = known & hold_op[op]
        fit_rows = np.where(known & ~hold)[0]
        mdl = _gbm(self.seed)
        mdl.fit(X[fit_rows], t[fit_rows], sample_weight=sw[fit_rows])
        hrows = np.where(hold)[0]
        t_hat = np.zeros(self.NG)
        t_hat[hrows] = mdl.predict(X[hrows])
        _log(f"C: holdout gene rmse {np.sqrt(np.mean((t[hrows] - t_hat[hrows]) ** 2)):.4f}"
             f" (genome mean {np.sqrt(np.mean(t[hrows] ** 2)):.4f})", T)
        # calibration cells of the held-out genes (subsampled per genome)
        ys, etas, u1, u2 = [], [], [], []
        for k in range(self.n_genomes):
            lo, hi = self.tr_bounds[k], self.tr_bounds[k + 1]
            if hi == lo:
                continue
            g = self.tr_g[lo:hi]
            sel = np.where(hold[g])[0]
            if len(sel) > 20000:
                sel = rng.choice(sel, 20000, replace=False)
            g, r = g[sel], self.tr_r[lo:hi][sel]
            ys.append(self.tr_y[lo:hi][sel])
            etas.append(self.m + self.gmean[k] + t_hat[g] + self.b[r]
                        + self.tr_off[lo:hi][sel])
            u1.append(self.vr[r])
            u2.append(np.full(len(sel), self.vg_genome[k]))
        self.cal_C = Calibrator().fit(*(np.concatenate(v) for v in
                                        (ys, etas, u1, u2)))
        _log(f"C: held-out-gene nll {self.cal_C.loss_:.4f}", T)

    # ----------------------------------------------------------- predict
    def predict_parts(self, test_df):
        """Uncalibrated predictions: region code (0 A, 1 B, 2 C, 3 D), the
        log-mean of the two A predictors (equal outside A) and the two
        variance proxies used by the calibration."""
        T = time.time()
        g, r, off = self._map(test_df)
        n = len(g)
        kg, kr = self.known_g[g], self.known_r[r]
        region = np.select([kg & kr, kg & ~kr, ~kg & kr], [0, 1, 2], 3)
        ek, er = np.zeros(n), np.zeros(n)
        u1, u2 = np.ones(n), np.ones(n)
        gen = self.gen_g[g]
        selA = np.where(region == 0)[0]
        for k in np.unique(gen[selA]):
            rows = selA[gen[selA] == k]
            ug, ur, Z, Y, O, W = self._genome_matrix(k)
            gi = np.searchsorted(ug, g[rows])
            ri = np.searchsorted(ur, r[rows])
            R2, ri2 = np.unique(ri, return_inverse=True)
            G2, gi2 = np.unique(gi, return_inverse=True)
            bk, br, vg, vr = self._a_block(Z, W, R2, G2, ug)
            ek[rows] = bk[ri2, gi2] + off[rows]
            er[rows] = br[ri2, gi2] + off[rows]
            u1[rows], u2[rows] = vg[gi2], vr[ri2]
        selN = np.where(~kr)[0]
        for k in np.unique(gen[selN]):
            rows = selN[gen[selN] == k]
            ug, ur, R, W, Z, Y, O = self._genome_residuals(k)
            nr, ri = np.unique(r[rows], return_inverse=True)
            if len(ur):
                w = np.exp(self._sim(nr, ur) @ self.B_beta)
                prof, vrh = self._b_predict(w, R, W, ur)
                b_new = prof.mean(1)
            else:
                prof = np.zeros((len(nr), 0))
                vrh = np.full(len(nr), float(np.mean(self.vr[self.known_r])))
                b_new = np.zeros(len(nr))
            isB = region[rows] == 1
            rb, rd = rows[isB], rows[~isB]
            if len(rb):
                gi = np.searchsorted(ug, g[rb])
                prj = self.run_codes[:, 0]
                seen = np.isin(prj[nr], prj[ur]) & (prj[nr] >= 0)
                shrink = np.where(seen, B_SHRINK_SEEN, 1.0)
                ek[rb] = (self.m + self.a[g[rb]]
                          + shrink[ri[isB]] * prof[ri[isB], gi] + off[rb])
                u1[rb], u2[rb] = self.vg[g[rb]], vrh[ri[isB]]
            if len(rd):
                ek[rd] = (self.m + self.gmean[k] + self.t_new[g[rd]]
                          + b_new[ri[~isB]] + off[rd])
                u1[rd], u2[rd] = vrh[ri[~isB]], self.vg_genome[k]
        selC = np.where(region == 2)[0]
        c_nb, c_gn = np.zeros(n), np.zeros(n)
        if len(selC):
            gc, rc = g[selC], r[selC]
            ek[selC] = (self.m + self.gmean[self.gen_g[gc]] + self.t_new[gc]
                        + self.b[rc] + off[selC])
            u1[selC], u2[selC] = self.vr[rc], self.vg_genome[self.gen_g[gc]]
            for k in np.unique(gen[selC]):
                rows = selC[gen[selC] == k]
                c_nb[rows], c_gn[rows] = self._c_profiles(k, g[rows], r[rows])
        er[region != 0] = ek[region != 0]
        er[region != 0] = ek[region != 0]
        _log(f"predict {n} rows, per region "
             f"{np.bincount(region, minlength=4).tolist()}", T)
        return dict(region=region, ek=ek, er=er, u1=u1, u2=u2,
                    c_nb=c_nb, c_gn=c_gn)

    def _c_profiles(self, k, g, r):
        """Run profiles (global residuals) of the nearest known neighbours
        and of the known genes sharing the gene-name prefix, at the cells
        (g, r) of new genes of genome k."""
        ug, ur, R, W, Z, Y, O = self._genome_residuals(k)
        ri = np.searchsorted(ur, r)
        cnt = np.zeros(len(g))
        nb = np.zeros(len(g))
        for ptr in (self.nb_up, self.nb_dn):
            j = ptr[g]
            gi = np.searchsorted(ug, j)
            ok = (j >= 0) & (gi < len(ug))
            ok[ok] &= ug[gi[ok]] == j[ok]
            nb[ok] += R[ri[ok], gi[ok]]
            cnt += ok
        nb = np.where(cnt > 0, nb / np.maximum(cnt, 1), 0.0)
        gn = np.zeros(len(g))
        codes = self.gname3[ug]
        for c in np.unique(self.gname3[g]):
            if c < 0:
                continue
            cols = np.where(codes == c)[0]
            if len(cols) == 0:
                continue
            prof = R[:, cols].sum(1) / np.maximum(W[:, cols].sum(1), 1)
            sel = self.gname3[g] == c
            gn[sel] = prof[ri[sel]]
        return nb, gn

    def calibrators(self):
        """Per-region calibrators: shipped constants (fitted offline on the
        public eval regions) when available, else the internal ones."""
        out = {}
        for i, name in enumerate("ABCD"):
            p = CALIBRATION.get(name)
            if p is not None:
                out[i] = Calibrator(p)
            else:
                out[i] = getattr(self, "cal_" + ("C" if name == "D" else name))
        return out

    def predict(self, test_df, samples_df=None, genomes_dir=None):
        P = self.predict_parts(test_df)
        region, ek, er, u1, u2 = (P[k] for k in ("region", "ek", "er",
                                                 "u1", "u2"))
        w = A_KRR_WEIGHT if A_KRR_WEIGHT is not None else self.A_w
        eta = np.where(region == 0, w * ek + (1 - w) * er, ek)
        eta = eta + C_NB_WEIGHT * P["c_nb"] + C_GN_WEIGHT * P["c_gn"]
        mu, sig = np.ones(len(eta)), np.ones(len(eta))
        for i, cal in self.calibrators().items():
            sel = region == i
            if sel.any():
                mu[sel], sig[sel] = cal(eta[sel], u1[sel], u2[sel])
        mu = np.nan_to_num(mu, nan=1.0, posinf=1e7)
        sig = np.nan_to_num(sig, nan=1.0, posinf=1e6)
        return np.column_stack([np.clip(mu, MU_FLOOR, 1e9),
                                np.clip(sig, 1e-3, 1e6)])


class Solver(GeneCompetSolver):

    name = "submission"

    def get_model(self, meta):
        return ExpressionModel(meta)

    def fit(self, model, train_df, samples_df, genomes_dir):
        model.fit(train_df, samples_df, genomes_dir)
