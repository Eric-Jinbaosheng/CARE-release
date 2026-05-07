#!/usr/bin/env python3
import argparse
import ast
import base64
import csv
import io
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageOps, ImageDraw

csv.field_size_limit(sys.maxsize)

ROOT = Path('<ANON_ROOT>/peking/smolvlm2_paper/ets_clean')
LMU = Path('<ANON_ROOT>/LMUData')

NS_MAIN = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
NS_REL = '{http://schemas.openxmlformats.org/package/2006/relationships}'
NS_DOC_REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'


def norm_text(s: str) -> str:
    if s is None:
        return ''
    s = str(s).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = s.replace(',', '')
    return s


def try_float(s: str):
    s = norm_text(s)
    if s == '':
        return None
    try:
        return float(s)
    except Exception:
        return None


def is_correct_simple(pred: str, gt: str) -> bool:
    p = norm_text(pred)
    g = norm_text(gt)
    # list GT string support
    if g.startswith('[') and g.endswith(']'):
        try:
            arr = ast.literal_eval(gt)
            if isinstance(arr, list):
                for item in arr:
                    if norm_text(item) == p:
                        return True
                # numeric fallback in list
                pf = try_float(pred)
                if pf is not None:
                    for item in arr:
                        gf = try_float(str(item))
                        if gf is not None and abs(pf - gf) < 1e-9:
                            return True
                return False
        except Exception:
            pass

    pf = try_float(pred)
    gf = try_float(gt)
    if pf is not None and gf is not None:
        return abs(pf - gf) < 1e-9

    return p == g


def read_xlsx_rows(path: Path):
    with zipfile.ZipFile(path) as zf:
        shared = []
        if 'xl/sharedStrings.xml' in zf.namelist():
            root = ET.fromstring(zf.read('xl/sharedStrings.xml'))
            for si in root.findall(f'{NS_MAIN}si'):
                txt = []
                for t in si.iter(f'{NS_MAIN}t'):
                    txt.append(t.text or '')
                shared.append(''.join(txt))

        wb = ET.fromstring(zf.read('xl/workbook.xml'))
        sheet = wb.find(f'{NS_MAIN}sheets/{NS_MAIN}sheet')
        rid = sheet.attrib.get(NS_DOC_REL + 'id')

        rels = ET.fromstring(zf.read('xl/_rels/workbook.xml.rels'))
        target = None
        for r in rels.findall(f'{NS_REL}Relationship'):
            if r.attrib.get('Id') == rid:
                target = r.attrib.get('Target')
                break
        if target is None:
            return []
        if not target.startswith('xl/'):
            target = 'xl/' + target

        sh = ET.fromstring(zf.read(target))
        rows = sh.findall(f'{NS_MAIN}sheetData/{NS_MAIN}row')

        def cell_value(c):
            t = c.attrib.get('t')
            v = c.find(f'{NS_MAIN}v')
            if v is None:
                is_el = c.find(f'{NS_MAIN}is')
                if is_el is not None:
                    ts = [x.text or '' for x in is_el.iter(f'{NS_MAIN}t')]
                    return ''.join(ts)
                return ''
            x = v.text or ''
            if t == 's':
                try:
                    return shared[int(x)]
                except Exception:
                    return x
            return x

        out = []
        if not rows:
            return out
        header = [cell_value(c) for c in rows[0].findall(f'{NS_MAIN}c')]
        for rr in rows[1:]:
            vals = [cell_value(c) for c in rr.findall(f'{NS_MAIN}c')]
            if len(vals) < len(header):
                vals += [''] * (len(header) - len(vals))
            row = {header[i]: vals[i] for i in range(len(header))}
            out.append(row)
        return out


def build_pred_map(xlsx_path: Path):
    rows = read_xlsx_rows(xlsx_path)
    mp = {}
    for r in rows:
        idx = str(r.get('index', '')).strip()
        if idx:
            mp[idx] = r
    return mp


def load_tsv_rows(tsv_path: Path, target_ids):
    out = {}
    with open(tsv_path, 'r', encoding='utf-8') as fh:
        rd = csv.DictReader(fh, delimiter='\t')
        for r in rd:
            idx = str(r.get('index', '')).strip()
            if idx in target_ids:
                out[idx] = r
                if len(out) == len(target_ids):
                    break
    return out


def decode_image_to(path: Path, image_field: str):
    cand = image_field
    if cand is None:
        return False, 'image field missing'
    cand = str(cand)
    if cand.startswith('[') and cand.endswith(']'):
        try:
            arr = ast.literal_eval(cand)
            if isinstance(arr, list) and arr:
                cand = arr[0]
        except Exception:
            pass
    try:
        raw = base64.b64decode(cand)
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return True, ''
    except Exception as e:
        return False, f'base64 decode failed: {e}'


def lookup_row_by_index(tsv_path: Path, target_index: str):
    with open(tsv_path, 'r', encoding='utf-8') as fh:
        rd = csv.DictReader(fh, delimiter='\t')
        for r in rd:
            if str(r.get('index', '')).strip() == target_index:
                return r
    return None


def resolve_image_reference_chain(tsv_path: Path, row: dict, max_hops: int = 5):
    cur = row
    visited = set()
    for _ in range(max_hops):
        img = str(cur.get('image', '') or '').strip()
        if not img:
            return cur
        # list-like field
        if img.startswith('[') and img.endswith(']'):
            return cur
        # likely base64
        if len(img) > 64:
            return cur
        # likely index reference
        ref = img
        if ref in visited:
            return cur
        visited.add(ref)
        nxt = lookup_row_by_index(tsv_path, ref)
        if nxt is None:
            return cur
        cur = nxt
    return cur


def ensure_image(benchmark: str, row: dict, image_out_path: Path, tsv_path: Path):
    # resolve canonical source path first
    src = None
    idx = str(row.get('index', '')).strip()
    if benchmark == 'ocrvqa':
        im_name = str(row.get('image_path', '')).strip()
        if im_name:
            src = LMU / 'images' / 'OCRVQA' / im_name
    elif benchmark == 'textvqa':
        src = LMU / 'images' / 'TextVQA_VAL' / f'{idx}.jpg'
        if not src.exists():
            ref = str(row.get('image', '')).strip()
            if ref.isdigit():
                src2 = LMU / 'images' / 'TextVQA_VAL' / f'{ref}.jpg'
                if src2.exists():
                    src = src2
    elif benchmark == 'ocrbench':
        src = LMU / 'images' / 'OCRBench' / f'{idx}.jpg'
    elif benchmark == 'chartqa':
        src = LMU / 'images' / 'ChartQA_TEST' / f'{idx}.jpg'

    if src is not None and src.exists():
        image_out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(src).convert('RGB').save(image_out_path)
        return str(src), ''

    # fallback decode from tsv image field
    use_row = resolve_image_reference_chain(tsv_path, row)
    ok, msg = decode_image_to(image_out_path, use_row.get('image'))
    if ok:
        return str(image_out_path), ''

    reason = f'image missing; tried {src}; {msg}; check annotation {benchmark.upper()}_*.tsv'
    return '', reason


def make_contact_sheet(image_paths, out_path: Path, cols=4, cell_w=360, cell_h=260):
    if not image_paths:
        return False
    rows = (len(image_paths) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * cell_w, rows * cell_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for i, (title, p) in enumerate(image_paths):
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w, r * cell_h
        try:
            im = Image.open(p).convert('RGB')
            im.thumbnail((cell_w - 10, cell_h - 40))
            px = x0 + (cell_w - im.width) // 2
            py = y0 + 5
            canvas.paste(im, (px, py))
            draw.text((x0 + 5, y0 + cell_h - 28), title, fill=(20, 20, 20))
        except Exception:
            draw.rectangle([x0 + 5, y0 + 5, x0 + cell_w - 5, y0 + cell_h - 35], outline=(180, 0, 0), width=2)
            draw.text((x0 + 8, y0 + 8), f'BROKEN: {title}', fill=(180, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=str(ROOT / 'paper_figures' / 'cases'))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # fixed case list
    cases = [
        ('textvqa', '34777', 'textvqa_34777', 'textvqa_rescue'),
        ('textvqa', '35677', 'textvqa_35677', 'textvqa_rescue'),
        ('textvqa', '38402', 'textvqa_cf_38402', 'textvqa_cf_rescue'),
        ('ocrvqa', '1610463633_0', 'ocrvqa_1610463633_0', 'ocrvqa_rescue'),
        ('ocrvqa', '146831002X_4', 'ocrvqa_146831002X_4', 'ocrvqa_rescue'),
        ('ocrvqa', '964920549_0', 'ocrvqa_964920549_0', 'ocrvqa_rescue'),
        ('ocrbench', '57', 'ocrbench_57', 'ocrbench_rescue'),
        ('ocrbench', '138', 'ocrbench_138', 'ocrbench_rescue'),
        ('ocrbench', '409', 'ocrbench_409', 'ocrbench_rescue'),
        ('chartqa', '30', 'chartqa_30', 'chartqa_ungated_candidate'),
        ('chartqa', '37', 'chartqa_37', 'chartqa_ungated_candidate'),
        ('chartqa', '40', 'chartqa_40', 'chartqa_ungated_candidate'),
    ]

    # load official prediction maps
    pred_paths = {
        'textvqa': {
            'ttaug': ROOT / 'benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_textvqa/TTAugClassical_SmolVLM2_2B/T20260425_G8433322c/TTAugClassical_SmolVLM2_2B_TextVQA_VAL.xlsx',
            'asca': ROOT / 'benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_regen_textvqa/V91NoCF_SmolVLM2_2B/T20260503_G8433322c/V91NoCF_SmolVLM2_2B_TextVQA_VAL.xlsx',
            'cf': ROOT / 'benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_routed_textvqa/V91CF3Routed_SmolVLM2_2B/T20260430_G8433322c/V91CF3Routed_SmolVLM2_2B_TextVQA_VAL.xlsx',
            'ungated': ROOT / 'paper_neurips2026_artifacts/ablations/groupB_allcf_4bench_20260504_official/official_eval_inputs_groupB/textvqa/all_cf_ungated/all_cf_ungated_TextVQA_VAL.xlsx',
            'tsv': LMU / 'TextVQA_VAL.tsv',
        },
        'ocrvqa': {
            'ttaug': ROOT / 'benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_ocrvqa/TTAugClassical_SmolVLM2_2B/T20260425_G8433322c/TTAugClassical_SmolVLM2_2B_OCRVQA_TEST.xlsx',
            'asca': ROOT / 'benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_regen_ocrvqa/V91NoCF_SmolVLM2_2B/T20260503_G8433322c/V91NoCF_SmolVLM2_2B_OCRVQA_TEST.xlsx',
            'cf': ROOT / 'benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_routed_ocrvqa/V91CF3Routed_SmolVLM2_2B/T20260504_G8433322c/V91CF3Routed_SmolVLM2_2B_OCRVQA_TEST.xlsx',
            'ungated': ROOT / 'paper_neurips2026_artifacts/ablations/groupB_allcf_3bench_20260504_sep/ocrvqa/official_eval_inputs_groupB/ocrvqa/all_cf_ungated/all_cf_ungated_OCRVQA_TEST.xlsx',
            'tsv': LMU / 'OCRVQA_TEST.tsv',
        },
        'ocrbench': {
            'ttaug': ROOT / 'benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_ocrbench/TTAugClassical_SmolVLM2_2B/T20260425_G8433322c/TTAugClassical_SmolVLM2_2B_OCRBench.xlsx',
            'asca': ROOT / 'benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_regen_ocrbench/V91NoCF_SmolVLM2_2B/T20260503_G8433322c/V91NoCF_SmolVLM2_2B_OCRBench.xlsx',
            'cf': ROOT / 'benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_routed_ocrbench/V91CF3Routed_SmolVLM2_2B/T20260504_G8433322c/V91CF3Routed_SmolVLM2_2B_OCRBench.xlsx',
            'ungated': ROOT / 'paper_neurips2026_artifacts/ablations/groupB_allcf_3bench_20260504_sep/ocrbench/official_eval_inputs_groupB/ocrbench/all_cf_ungated/all_cf_ungated_OCRBench.xlsx',
            'tsv': LMU / 'OCRBench.tsv',
        },
        'chartqa': {
            'ttaug': ROOT / 'benchmark_results/n_samples_1000/test_config_smolvlm2_paper_ttaug_classical_chartqa/TTAugClassical_SmolVLM2_2B/T20260425_G8433322c/TTAugClassical_SmolVLM2_2B_ChartQA_TEST.xlsx',
            'asca': ROOT / 'benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_regen_chartqa/V91NoCF_SmolVLM2_2B/T20260503_G8433322c/V91NoCF_SmolVLM2_2B_ChartQA_TEST.xlsx',
            'cf': ROOT / 'benchmark_results/n_samples_1000/test_config_smolvlm2_v91_cf3_routed_chartqa/V91CF3Routed_SmolVLM2_2B/T20260504_G8433322c/V91CF3Routed_SmolVLM2_2B_ChartQA_TEST.xlsx',
            'ungated': ROOT / 'paper_neurips2026_artifacts/ablations/groupB_allcf_3bench_20260504_sep/chartqa/official_eval_inputs_groupB/chartqa/all_cf_ungated/all_cf_ungated_ChartQA_TEST.xlsx',
            'tsv': LMU / 'ChartQA_TEST.tsv',
        }
    }

    # maps per bench
    pred_maps = {}
    for b, cfg in pred_paths.items():
        pred_maps[b] = {}
        for k in ['ttaug', 'asca', 'cf', 'ungated']:
            p = cfg[k]
            if p.exists():
                pred_maps[b][k] = build_pred_map(p)
            else:
                pred_maps[b][k] = {}

    # load raw tsv rows only for needed ids per benchmark
    ids_by_bench = {}
    for b, sid, _, _ in cases:
        ids_by_bench.setdefault(b, set()).add(str(sid))

    ann = {}
    for b, ids in ids_by_bench.items():
        ann[b] = load_tsv_rows(pred_paths[b]['tsv'], ids)

    # parse case types from oracle changed table for ocr tasks
    changed_type = {}
    oracle = ROOT / 'paper_neurips2026_artifacts/tables/oracle_gap_changed_examples_n1000.csv'
    if oracle.exists():
        for r in csv.DictReader(open(oracle, encoding='utf-8')):
            key = (r.get('benchmark', ''), str(r.get('sample_id', '')))
            changed_type[key] = r.get('type', '')

    # parse curated textvqa case type from qualitative compact
    qual_type = {}
    qual = ROOT / 'paper_neurips2026_artifacts/tables/qualitative_examples_compact.csv'
    if qual.exists():
        for r in csv.DictReader(open(qual, encoding='utf-8')):
            sid = str(r.get('SampleID', '')).strip()
            t = r.get('Type', '')
            if sid and t and sid not in qual_type:
                qual_type[sid] = t
    qual_cf = {}
    if qual.exists():
        for r in csv.DictReader(open(qual, encoding='utf-8')):
            sid = str(r.get('SampleID', '')).strip()
            if sid:
                qual_cf[sid] = r.get('CF', '')

    out_rows = []
    image_items = []
    missing = []

    for bench, sid, name_stub, default_case_type in cases:
        sid = str(sid)
        row = ann.get(bench, {}).get(sid)
        if row is None:
            missing.append((bench, sid, f'missing annotation row in {pred_paths[bench]["tsv"]}'))
            continue

        q = row.get('question', '')
        gt = row.get('answer', '')

        def gp(kind):
            rr = pred_maps[bench].get(kind, {}).get(sid, {})
            return rr.get('prediction', '') if rr else ''

        ttaug_pred = gp('ttaug')
        asca_pred = gp('asca')
        cf_pred = gp('cf')
        ung_pred = gp('ungated')

        # fallback for textvqa cf fields from curated table (if empty)
        if bench == 'textvqa':
            if not cf_pred:
                rr = pred_maps[bench].get('cf', {}).get(sid, {})
                cf_pred = rr.get('prediction', '')
            if sid in qual_cf and qual_cf[sid]:
                cf_pred = qual_cf[sid]

        # ChartQA final-selection rule for ungated candidates
        case_type = default_case_type
        if bench == 'chartqa':
            asca_ok = is_correct_simple(asca_pred, gt)
            ung_wrong = (not is_correct_simple(ung_pred, gt)) if ung_pred != '' else False
            if asca_ok and ung_wrong:
                case_type = 'chartqa_ungated_harm_confirmed'
            else:
                case_type = 'chartqa_ungated_candidate_rejected'

        out_img = out_dir / f'{name_stub}.jpg'
        img_path, err = ensure_image(bench, row, out_img, pred_paths[bench]['tsv'])
        if err:
            missing.append((bench, sid, err))
            img_path = ''

        if out_img.exists():
            image_items.append((f'{bench}:{sid}', out_img))

        # enrich case type from changed tables when available
        if bench in ('ocrvqa', 'ocrbench'):
            ctype = changed_type.get((bench, sid), '')
            if ctype:
                case_type = f'{default_case_type}|{ctype}'
        if bench == 'textvqa':
            qt = qual_type.get(sid, '')
            if qt:
                case_type = f'{default_case_type}|{qt}'

        out_rows.append({
            'benchmark': bench,
            'sample_id': sid,
            'image_path': img_path,
            'question': q,
            'ground_truth': gt,
            'ttaug_pred': ttaug_pred,
            'asca_pred': asca_pred,
            'cf_pred': cf_pred,
            'ungated_cf_pred': ung_pred,
            'case_type': case_type,
        })

    # apply final filter for chartqa request (only keep confirmed)
    final_rows = []
    chartqa_checks = []
    for r in out_rows:
        if r['benchmark'] == 'chartqa':
            chartqa_checks.append({
                'sample_id': r['sample_id'],
                'ground_truth': r['ground_truth'],
                'asca_pred': r['asca_pred'],
                'ungated_cf_pred': r['ungated_cf_pred'],
                'asca_correct': int(is_correct_simple(r['asca_pred'], r['ground_truth'])),
                'ungated_wrong': int(not is_correct_simple(r['ungated_cf_pred'], r['ground_truth']) if r['ungated_cf_pred'] != '' else 0),
                'selected_final': int(r['case_type'] == 'chartqa_ungated_harm_confirmed'),
            })
            if r['case_type'] != 'chartqa_ungated_harm_confirmed':
                continue
        final_rows.append(r)

    all_csv_path = out_dir / 'case_manifest_all_requested.csv'
    with open(all_csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['benchmark','sample_id','image_path','question','ground_truth','ttaug_pred','asca_pred','cf_pred','ungated_cf_pred','case_type'])
        w.writeheader()
        w.writerows(out_rows)

    csv_path = out_dir / 'case_manifest.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['benchmark','sample_id','image_path','question','ground_truth','ttaug_pred','asca_pred','cf_pred','ungated_cf_pred','case_type'])
        w.writeheader()
        w.writerows(final_rows)

    chart_check_path = out_dir / 'chartqa_ungated_candidate_check.csv'
    with open(chart_check_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id','ground_truth','asca_pred','ungated_cf_pred','asca_correct','ungated_wrong','selected_final'])
        w.writeheader()
        w.writerows(chartqa_checks)

    miss_path = out_dir / 'case_missing_reasons.txt'
    with open(miss_path, 'w', encoding='utf-8') as f:
        if not missing:
            f.write('None\n')
        else:
            for b, sid, reason in missing:
                f.write(f'{b},{sid},{reason}\n')

    sheet_path = out_dir / 'contact_sheet.jpg'
    make_contact_sheet(image_items, sheet_path)

    print('CSV:', csv_path)
    print('CSV_ALL:', all_csv_path)
    print('CHARTQA_CHECK:', chart_check_path)
    print('CONTACT:', sheet_path)
    print('MISSING:', miss_path)
    print('TOTAL_ROWS_FINAL:', len(final_rows))
    print('TOTAL_IMAGES:', len(image_items))

if __name__ == '__main__':
    main()
