#!/usr/bin/env python3
import argparse
import ast
import json
import math
import statistics
import string
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
PUNCT = str.maketrans("", "", string.punctuation)


def norm(s):
    if s is None:
        return ""
    return str(s).strip().lower().translate(PUNCT).strip()


def col_to_idx(col):
    n = 0
    for ch in col:
        if ch.isalpha():
            n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def read_xlsx_rows(path):
    z = zipfile.ZipFile(path)
    sst = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall("a:si", NS):
            sst.append("".join((t.text or "") for t in si.findall(".//a:t", NS)))

    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rel = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    sheet = wb.find("a:sheets/a:sheet", NS)
    rid = sheet.attrib[
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    ]
    target = None
    for r in rel:
        if r.attrib.get("Id") == rid:
            target = r.attrib.get("Target")
            break
    sh = ET.fromstring(z.read("xl/" + target))
    rows = []
    for row in sh.findall("a:sheetData/a:row", NS):
        cells = {}
        for c in row.findall("a:c", NS):
            ref = c.attrib.get("r", "A1")
            col = "".join(ch for ch in ref if ch.isalpha())
            idx = col_to_idx(col)
            t = c.attrib.get("t")
            v = c.find("a:v", NS)
            val = ""
            if t == "s" and v is not None:
                i = int(v.text)
                val = sst[i] if 0 <= i < len(sst) else ""
            elif t == "inlineStr":
                x = c.find("a:is/a:t", NS)
                val = x.text if x is not None else ""
            elif v is not None:
                val = v.text or ""
            cells[idx] = val
        if cells:
            m = max(cells)
            arr = [""] * (m + 1)
            for i, v in cells.items():
                arr[i] = v
            rows.append(arr)
    header = rows[0]
    out = []
    for r in rows[1:]:
        out.append({header[i]: (r[i] if i < len(r) else "") for i in range(len(header))})
    return out


def parse_gt_set(ans_raw):
    try:
        ans = ast.literal_eval(ans_raw)
        if not isinstance(ans, list):
            ans = [str(ans)]
    except Exception:
        ans = [ans_raw]
    return {norm(a) for a in ans if norm(a)}


def load_general_scores(scored_top):
    out = {}
    if not isinstance(scored_top, list):
        return out
    for t in scored_top:
        if not isinstance(t, (list, tuple)) or len(t) < 3:
            continue
        out[norm(t[0])] = float(t[2])
    return out


def auc_pairwise(score_map, correct_set):
    cands = list(score_map.keys())
    pos = [c for c in cands if c in correct_set]
    neg = [c for c in cands if c not in correct_set]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            sp = score_map[p]
            sn = score_map[n]
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
            total += 1
    return wins / total if total else None


def build_scores(cands, logp_pos, logp_rel, logp_ctrl):
    s = {}
    for c in cands:
        lp_pos = float(logp_pos.get(c, -100.0))
        lp_rel = float(logp_rel.get(c, -100.0))
        lp_ctrl = float(logp_ctrl.get(c, -100.0))
        s1 = lp_ctrl - lp_rel
        s2 = lp_rel - lp_ctrl
        s3 = (lp_pos - lp_rel) - (lp_pos - lp_ctrl)
        # In current implementation logp_* are already per-token averages.
        s4 = s1
        s5 = s1
        s[c] = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
    return s


def score_winner(score_map):
    return max(score_map.keys(), key=lambda k: score_map[k]) if score_map else ""


def run(args):
    xlsx_rows = read_xlsx_rows(Path(args.xlsx))
    recs = [
        json.loads(x)
        for x in Path(args.diag).read_text(errors="ignore").splitlines()
        if x.strip()
    ]
    n = min(len(xlsx_rows), len(recs))
    xlsx_rows = xlsx_rows[:n]
    recs = recs[:n]

    formulas = ["s1", "s2", "s3", "s4", "s5"]
    stats = {
        f: {
            "disagree": 0,
            "rescue": 0,
            "harm": 0,
            "neutral": 0,
            "always_ok": 0,
            "nocf_ok": 0,
            "bestof_ok": 0,
            "margins": [],
            "samples": [],
        }
        for f in formulas
    }
    aucs = {f: [] for f in formulas}
    mask_split = {f: defaultdict(lambda: Counter()) for f in formulas}
    rel_ctrl_sanity = Counter()
    rel_ctrl_vals = []

    for row, r in zip(xlsx_rows, recs):
        gt = parse_gt_set(row.get("answer", ""))
        no_cf = norm(r.get("no_cf_winner") or r.get("final_answer") or r.get("base_answer"))
        logp_pos = {norm(k): float(v) for k, v in (r.get("logp_pos") or {}).items()}
        logp_rel = {norm(k): float(v) for k, v in (r.get("logp_rel") or {}).items()}
        logp_ctrl = {norm(k): float(v) for k, v in (r.get("logp_ctrl") or {}).items()}
        cands = [norm(c) for c in (r.get("candidate_list") or list(logp_pos.keys())) if norm(c)]
        cands = sorted(set(cands))
        if not cands:
            continue

        scores = build_scores(cands, logp_pos, logp_rel, logp_ctrl)

        # rel/control sanity on GT candidates
        gt_in_cands = [c for c in cands if c in gt]
        for c in gt_in_cands:
            lp_rel = logp_rel.get(c, -100.0)
            lp_ctrl = logp_ctrl.get(c, -100.0)
            rel_ctrl_vals.append(lp_ctrl - lp_rel)
            if lp_ctrl > lp_rel:
                rel_ctrl_sanity["ctrl_gt_rel"] += 1
            else:
                rel_ctrl_sanity["ctrl_le_rel"] += 1

        for f in formulas:
            fmap = {c: scores[c][f] for c in cands}
            cf_w = score_winner(fmap)
            no_ok = no_cf in gt if no_cf else False
            cf_ok = cf_w in gt if cf_w else False
            margin = fmap.get(cf_w, 0.0) - fmap.get(no_cf, 0.0)
            stats[f]["margins"].append(margin)
            stats[f]["samples"].append((margin, no_ok, cf_ok, cf_w, no_cf, r, row))

            if no_ok:
                stats[f]["nocf_ok"] += 1
            if cf_ok:
                stats[f]["always_ok"] += 1
            if no_ok or cf_ok:
                stats[f]["bestof_ok"] += 1

            if cf_w and no_cf and cf_w != no_cf:
                stats[f]["disagree"] += 1
                if cf_ok and not no_ok:
                    stats[f]["rescue"] += 1
                elif no_ok and not cf_ok:
                    stats[f]["harm"] += 1
                else:
                    stats[f]["neutral"] += 1

            mkey = f"{r.get('mask_type','unknown')}|{r.get('mask_quality','unknown')}"
            ms = mask_split[f][mkey]
            ms["n"] += 1
            if cf_ok:
                ms["cf_winner_ok"] += 1
            if cf_w and no_cf and cf_w != no_cf:
                if cf_ok and not no_ok:
                    ms["rescue"] += 1
                elif no_ok and not cf_ok:
                    ms["harm"] += 1
                else:
                    ms["neutral"] += 1

            a = auc_pairwise(fmap, gt)
            if a is not None:
                aucs[f].append(a)

    print(f"N={n}")
    print("== Score Formula Table ==")
    for f in formulas:
        st = stats[f]
        disagree = st["disagree"]
        rescue = st["rescue"]
        harm = st["harm"]
        net = rescue - harm
        always_acc = st["always_ok"] / n
        nocf_acc = st["nocf_ok"] / n
        bestof_acc = st["bestof_ok"] / n
        rr = rescue / max(1, harm)
        print(
            f"{f}: always_acc={always_acc:.4f} nocf_acc={nocf_acc:.4f} "
            f"bestof={bestof_acc:.4f} bestof_delta={bestof_acc-nocf_acc:+.4f} "
            f"disagree={disagree} rescue={rescue} harm={harm} net={net:+d} r:h={rr:.2f}"
        )

        # threshold search on margin
        uniq = sorted(set(m for m in st["margins"]))
        best_acc = -1.0
        best_t = None
        best_r = best_h = 0
        for t in uniq:
            ok = 0
            r_cnt = 0
            h_cnt = 0
            for margin, no_ok, cf_ok, _, _, _, _ in st["samples"]:
                use_cf = margin > t
                pred_ok = cf_ok if use_cf else no_ok
                if pred_ok:
                    ok += 1
                if use_cf and cf_ok and not no_ok:
                    r_cnt += 1
                if use_cf and no_ok and not cf_ok:
                    h_cnt += 1
            acc = ok / n
            if acc > best_acc:
                best_acc = acc
                best_t = t
                best_r, best_h = r_cnt, h_cnt
        print(
            f"  best_margin_threshold={best_t:.6f} threshold_acc={best_acc:.4f} "
            f"delta_vs_nocf={best_acc-nocf_acc:+.4f} switch_rescue={best_r} switch_harm={best_h}"
        )

        bins = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 999.0]
        bcnt = Counter()
        for margin, no_ok, cf_ok, _, _, _, _ in st["samples"]:
            if margin <= 0:
                continue
            for b0, b1 in zip(bins[:-1], bins[1:]):
                if b0 <= margin < b1:
                    key = f"[{b0},{b1})"
                    bcnt[(key, "n")] += 1
                    if cf_ok and not no_ok:
                        bcnt[(key, "rescue")] += 1
                    elif no_ok and not cf_ok:
                        bcnt[(key, "harm")] += 1
                    break
        print("  margin_bins:")
        for b0, b1 in zip(bins[:-1], bins[1:]):
            key = f"[{b0},{b1})"
            nbin = bcnt[(key, "n")]
            if nbin == 0:
                continue
            print(
                f"    {key}: n={nbin} rescue={bcnt[(key,'rescue')]} harm={bcnt[(key,'harm')]}"
            )

    print("== AUC Table (correct vs wrong candidates) ==")
    for f in formulas:
        arr = aucs[f]
        mean_auc = statistics.mean(arr) if arr else float("nan")
        print(f"{f}: auc={mean_auc:.4f} n_samples={len(arr)}")

    print("== Mask Split (by s1 winner) ==")
    ms = mask_split["s1"]
    for k, c in sorted(ms.items(), key=lambda kv: -kv[1]["n"]):
        nmask = c["n"]
        acc = c["cf_winner_ok"] / max(1, nmask)
        print(
            f"{k}: n={nmask} cf_winner_acc={acc:.4f} rescue={c['rescue']} harm={c['harm']} neutral={c['neutral']}"
        )

    print("== Rel/Control Sanity (GT candidates) ==")
    tot = rel_ctrl_sanity["ctrl_gt_rel"] + rel_ctrl_sanity["ctrl_le_rel"]
    if tot:
        print(
            f"logP_ctrl(gt) > logP_rel(gt): {rel_ctrl_sanity['ctrl_gt_rel']}/{tot} "
            f"({rel_ctrl_sanity['ctrl_gt_rel']/tot:.4f})"
        )
        print(
            f"mean(logP_ctrl-logP_rel on gt)={statistics.mean(rel_ctrl_vals):.4f} "
            f"median={statistics.median(rel_ctrl_vals):.4f}"
        )
    else:
        print("No samples with GT candidate present.")

    print("== Top Rescue / Harm Examples (s1) ==")
    rescues = []
    harms = []
    for margin, no_ok, cf_ok, cf_w, no_cf, r, row in stats["s1"]["samples"]:
        if cf_w == no_cf:
            continue
        rec = {
            "sample_id": r.get("sample_id"),
            "question": row.get("question", ""),
            "image": row.get("image_path", ""),
            "gt": row.get("answer", ""),
            "no_cf": no_cf,
            "cf": cf_w,
            "margin": margin,
            "mask": r.get("mask_type", ""),
            "quality": r.get("mask_quality", ""),
            "space": r.get("answer_space", ""),
            "s1_no": (r.get("cf_score_raw", {}) or {}).get(no_cf, None),
            "s1_cf": (r.get("cf_score_raw", {}) or {}).get(cf_w, None),
            "logp_pos_no": (r.get("logp_pos", {}) or {}).get(no_cf, None),
            "logp_rel_no": (r.get("logp_rel", {}) or {}).get(no_cf, None),
            "logp_ctrl_no": (r.get("logp_ctrl", {}) or {}).get(no_cf, None),
            "logp_pos_cf": (r.get("logp_pos", {}) or {}).get(cf_w, None),
            "logp_rel_cf": (r.get("logp_rel", {}) or {}).get(cf_w, None),
            "logp_ctrl_cf": (r.get("logp_ctrl", {}) or {}).get(cf_w, None),
        }
        if cf_ok and not no_ok:
            rescues.append(rec)
        elif no_ok and not cf_ok:
            harms.append(rec)
    rescues = sorted(rescues, key=lambda x: -(x["margin"] or 0.0))[:10]
    harms = sorted(harms, key=lambda x: -(x["margin"] or 0.0))[:10]
    print("-- RESCUE --")
    for x in rescues:
        print(json.dumps(x, ensure_ascii=False))
    print("-- HARM --")
    for x in harms:
        print(json.dumps(x, ensure_ascii=False))

    print("== CF as Veto Policy (diagnostic) ==")
    # Policy A: switch only if margin>tau and general score close
    nocf_acc = stats["s1"]["nocf_ok"] / n
    candidates_tau = sorted(set(m for m in stats["s1"]["margins"]))
    best = (-1.0, None, 0, 0)
    for tau in candidates_tau:
        ok = 0
        r_cnt = 0
        h_cnt = 0
        for margin, no_ok, cf_ok, cf_w, no_cf, r, _ in stats["s1"]["samples"]:
            g = load_general_scores(r.get("scored_top"))
            no_g = g.get(no_cf, None)
            cf_g = g.get(cf_w, None)
            close = (
                no_g is not None
                and cf_g is not None
                and (cf_g >= no_g - args.general_gap)
            )
            use_cf = (margin > tau) and close
            pred_ok = cf_ok if use_cf else no_ok
            if pred_ok:
                ok += 1
            if use_cf and cf_ok and not no_ok:
                r_cnt += 1
            if use_cf and no_ok and not cf_ok:
                h_cnt += 1
        acc = ok / n
        if acc > best[0]:
            best = (acc, tau, r_cnt, h_cnt)
    print(
        f"policyA(best): acc={best[0]:.4f} delta_vs_nocf={best[0]-nocf_acc:+.4f} "
        f"tau={best[1]:.6f} rescue={best[2]} harm={best[3]}"
    )

    # Policy B: keep NoCF when NoCF is stable under relevant-drop, else allow switch on margin
    best_b = (-1.0, None, 0, 0)
    for tau in candidates_tau:
        ok = 0
        r_cnt = 0
        h_cnt = 0
        for margin, no_ok, cf_ok, cf_w, no_cf, r, _ in stats["s1"]["samples"]:
            lp_pos = (r.get("logp_pos", {}) or {}).get(no_cf, None)
            lp_rel = (r.get("logp_rel", {}) or {}).get(no_cf, None)
            stable = False
            if lp_pos is not None and lp_rel is not None:
                stable = (float(lp_pos) - float(lp_rel)) < args.rel_drop_eps
            use_cf = (not stable) and (margin > tau)
            pred_ok = cf_ok if use_cf else no_ok
            if pred_ok:
                ok += 1
            if use_cf and cf_ok and not no_ok:
                r_cnt += 1
            if use_cf and no_ok and not cf_ok:
                h_cnt += 1
        acc = ok / n
        if acc > best_b[0]:
            best_b = (acc, tau, r_cnt, h_cnt)
    print(
        f"policyB(best): acc={best_b[0]:.4f} delta_vs_nocf={best_b[0]-nocf_acc:+.4f} "
        f"tau={best_b[1]:.6f} rescue={best_b[2]} harm={best_b[3]}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--general-gap", type=float, default=0.15)
    ap.add_argument("--rel-drop-eps", type=float, default=0.05)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
