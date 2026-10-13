"""One frozen-weight APPS transfer study; reuse the sealed scorer and encoder."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlsplit
import qwen_fresh_20261011 as fresh
from rank_fusion_followup import ROOT, sha, write

development = fresh.development
previous = fresh.old.previous
OUT = ROOT / 'followup_results/qwen_apps_20261011'
EXEC = development.EXEC.parent / 'kaggle_qwen_apps_20261011'
PROTOCOL = ROOT / 'qwen_apps_protocol_20261011.md'
REVISION = '21e74ddf8de1a21436da12e3e653065c5213e9d1'
FILES = (
    ('train.jsonl', 107101272, '45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae'),
    ('test.jsonl', 1292436853, '5b003a65ac40feb47dd5eaec267a767a6fc435bdcfa68ff715fe869f948e760c'))
SLUG = 'code2hyp-qwen-apps-20261011'


def problem_url(value):
    u = urlsplit(value)
    if u.scheme not in ('http', 'https') or u.username or u.password:
        return None
    host = (u.hostname or '').removeprefix('www.')
    if host == 'codewars.com':
        match = re.fullmatch(r'/kata/([0-9a-fA-F]{24})(?:/(?:train|solutions))?(?:/python)?/?', u.path)
        if match:
            return 'https://codewars.com/kata/' + match[1].lower()
    elif host in ('open.kattis.com', 'kattis.com'):
        match = re.fullmatch(r'/problems/([a-zA-Z0-9._-]+)/?', u.path)
        if match:
            return 'https://open.kattis.com/problems/' + match[1]
    return None


def filter_previous(candidates, hashes):
    contaminated = {r['cluster_id'] for r in candidates
        if any(r[k] in h for k, h in zip(('d0', 'd1', 'd2'), hashes))}
    return [r for r in candidates if r['cluster_id'] not in contaminated], contaminated


def references(value):
    # APPS encodes missing reference solutions as an empty string.
    sources = [] if value == '' else json.loads(value)
    assert isinstance(sources, list) and all(isinstance(s, str) for s in sources)
    return sources


def prepare():
    if (OUT / 'design_receipt.json').exists():
        design = json.loads((OUT / 'design_receipt.json').read_text())
        development.verify_bindings(design)
        assert design['alpha'] == .10
        return design
    assert fresh.weight() == .10
    assert sha(development.OUT / 'weight_seal.json') == '72fb97afa09bbd166a9cfca33a6db8eab05867a842ed1cbb7f2acb8f2536822f'
    _, _, bindings = fresh.exclusions()
    bindings.update(json.loads((fresh.OUT / 'design_receipt.json').read_text())['bindings'])
    paths = [Path(__file__), PROTOCOL, ROOT / 'test_qwen_apps_20261011.py',
             development.EXEC / 'kernel/run.py', ROOT / 'qwen_calibration_encode_20261011.py',
             ROOT / 'crossed_encode_20261011.py', ROOT / 'qwen_external_encode_20261011.py',
             ROOT / 'coderank_kaggle_20261010.py', development.MODEL_RECEIPT]
    bindings.update({str(p): sha(p) for p in paths})
    design = {'created_before_apps_contents_utc': datetime.now(timezone.utc).isoformat(),
        'bindings': bindings, 'alpha': .10, 'dataset': 'codeparrot/apps', 'revision': REVISION,
        'files': FILES, 'max_problems': 200, 'min_problems': 30,
        'primary': 'selected_minus_dense_MAP8', 'bootstrap_seed': 20261014,
        'bootstrap_resamples': 20000, 'features_changed': False, 'java_test_accessed': False,
        'full_historical_fingerprint_inventory_available': False,
        'model_pretraining_overlap_known': False}
    development.verify_bindings(design)
    OUT.mkdir()
    write(OUT / 'design_receipt.json', design)
    return design


def select():
    prepare()
    if (OUT / 'selection_seal.json').exists():
        for name, digest in json.loads((OUT / 'selection_seal.json').read_text())['sha256'].items():
            assert sha(OUT / name) == digest, name
        return [json.loads(l) for l in (OUT / 'selected_programs.jsonl').read_text().splitlines()]
    assert not (OUT / 'selection_receipt.json').exists(), 'Do not replace an earlier selection'
    _, hashes, _ = fresh.exclusions()
    folder = OUT / 'input'
    folder.mkdir(exist_ok=True)
    candidates, attrition, platforms = [], Counter(), Counter()
    for name, size, digest in FILES:
        path = folder / name
        if not path.exists():
            partial = path.with_suffix('.part')
            url = f'https://huggingface.co/datasets/codeparrot/apps/resolve/{REVISION}/{name}?download=true'
            subprocess.run(['curl', '-fLsS', '--retry', '3', '--max-time', '900', '-o', str(partial), url], check=True)
            assert partial.stat().st_size == size and sha(partial) == digest, name
            partial.rename(path)
        assert path.stat().st_size == size and sha(path) == digest, name
        seen = set()
        for line in path.open():
            record = json.loads(line)
            assert record['id'] not in seen
            seen.add(record['id'])
            attrition['input_problem_records'] += 1
            url = problem_url(record['url'])
            if url is None:
                attrition['excluded_platform_or_unrecognised_url'] += 1
                continue
            platforms[urlsplit(url).hostname] += 1
            sources = references(record['solutions'])
            if not sources:
                attrition['no_reference_solutions'] += 1
            cluster = 'apps_' + hashlib.sha256(url.encode()).hexdigest()
            for source in sources:
                attrition['candidate_references'] += 1
                row, reason = previous.old.previous.candidate(source)
                if reason:
                    attrition[reason] += 1
                else:
                    candidates.append({**row, 'cluster_id': cluster, 'problem_url': url,
                        'apps_id': record['id'], 'apps_split': path.stem,
                        'difficulty': record['difficulty']})
        assert len(seen) == 5000, (name, len(seen))
        print('Eligibility processed', name, dict(attrition), flush=True)
    clean, contaminated = filter_previous(candidates, hashes)
    attrition['prior_fingerprint_problem'] = len(contaminated)
    attrition['prior_fingerprint_removed_references'] = len(candidates) - len(clean)
    _, eligible, removed = previous.choose(clean, 1)
    attrition.update(removed)
    target = min(200, eligible)
    rows, _, _ = previous.choose(clean, target) if target >= 30 else ([], eligible, None)
    receipt = {'eligible_problems': eligible, 'selected_problems': target if rows else 0,
        'attrition': dict(attrition), 'platform_problem_records': dict(platforms),
        'excluded_fingerprint_counts': list(map(len, hashes)),
        'excluded_problem_ids': sorted(contaminated), 'retrieval_scores_seen': False,
        'source_sha256': {n: h for n, _, h in FILES},
        'weight_seal_sha256': sha(development.OUT / 'weight_seal.json')}
    write(OUT / 'selection_receipt.json', receipt)
    if not rows:
        write(OUT / 'status.json', {'status': 'insufficient_eligible_problems_no_scores', 'problems': eligible})
        return []
    assert len(rows) == target * 16
    for k, h in zip(('d0', 'd1', 'd2'), hashes):
        assert len({r[k] for r in rows}) == len(rows) and not {r[k] for r in rows} & h
    for name, source in (('selected_programs.jsonl', True), ('metadata.jsonl', False)):
        with (OUT / name).open('x') as stream:
            for row in rows:
                stream.write(json.dumps({k: v for k, v in row.items() if source or k != 'source'}, sort_keys=True) + '\n')
    write(OUT / 'selection_seal.json', {'fixed_before_scores_utc': datetime.now(timezone.utc).isoformat(),
        'sha256': {n: sha(OUT / n) for n in ('selected_programs.jsonl', 'metadata.jsonl',
                                           'selection_receipt.json', 'design_receipt.json')}})
    print('SEALED', target, 'APPS problems;', len(rows), 'programs', flush=True)
    return rows


def prepare_run():
    rows = select()
    if not rows:
        return
    original = previous.OUT
    try:
        previous.OUT = OUT
        previous.features(rows)
    finally:
        previous.OUT = original
    if (EXEC / 'payload').exists():
        receipt = json.loads((EXEC / 'payload/qwen_weight_execution.json').read_text())
        assert receipt['design_sha256'] == sha(OUT / 'design_receipt.json')
        for name, digest in receipt['payload_sha256'].items():
            assert sha(EXEC / 'payload' / name) == digest, name
        return
    (EXEC / 'kernel').mkdir(parents=True)
    shutil.copy2(development.EXEC / 'kernel/run.py', EXEC / 'kernel/run.py')
    original = development.PROTOCOL
    try:
        development.PROTOCOL = PROTOCOL
        development.payload(OUT / 'selected_programs.jsonl', OUT / 'design_receipt.json', EXEC, SLUG)
    finally:
        development.PROTOCOL = original
    print('Payload ready; replace inherited dataset description/license with APPS provenance before upload', flush=True)


def run_evaluation(mode):
    # Reuse the exact scoring and independent-check implementation, not a fork.
    saved = fresh.OUT, fresh.EXEC, fresh.prepare, fresh.select
    try:
        fresh.OUT, fresh.EXEC, fresh.prepare, fresh.select = OUT, EXEC, prepare, select
        getattr(fresh, mode)()
    finally:
        fresh.OUT, fresh.EXEC, fresh.prepare, fresh.select = saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('design', 'prepare', 'evaluate', 'check'))
    mode = parser.parse_args().mode
    if mode == 'design':
        prepare()
    elif mode == 'prepare':
        prepare_run()
    else:
        run_evaluation(mode)
