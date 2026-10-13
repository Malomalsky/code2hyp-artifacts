"""Calibrate one Qwen weight on sealed old development queries only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import frequency_factorial_20261010 as old
from qwen_external_encode_20261011 import SETTINGS, validate
from rank_fusion_followup import ROOT, sha, write

OUT = ROOT/'followup_results/qwen_calibration_20261011'
EXEC = ROOT.parents[2]/'code2hyp_informatics_artifacts/local_execution/kaggle_qwen_calibration_20261011'
PROTOCOL = ROOT/'qwen_calibration_protocol_20261011.md'
MODEL_RECEIPT = EXEC.parent/'kaggle_cpret_external_20261011/output_v1/cpret_external_encoding/encoding_receipt.json'
GRID = old.GRID


def rerank(dense, structure, alpha, tie):
    assert 0 <= alpha <= 1 and len(dense) == len(structure) == len(tie)
    order = np.lexsort((tie, -dense))
    head = order[:100]
    mixed = (1-alpha)*dense[head]+alpha*structure[head]
    return np.concatenate((head[np.lexsort((tie[head],-mixed))], order[100:]))


def select_weight(table):
    assert table.ndim == 2 and len(table) and np.isfinite(table).all()
    return int(np.argmax(table.mean(axis=0)))


def metrics(order, positive):
    hit = positive[order]
    count = int(positive.sum())
    assert count == 8
    ap = float((np.cumsum(hit[:8])*hit[:8]/np.arange(1,9)).sum()/8)
    discount = 1/np.log2(np.arange(2,12))
    ndcg = float((hit[:10]*discount).sum()/discount[:8].sum())
    positions = np.flatnonzero(hit[:10])+1
    return [ap,ndcg,1/float(positions[0]) if len(positions) else 0.,float(hit[:100].sum()/8)]


def verify_bindings(design):
    for name,digest in design['bindings'].items(): assert sha(Path(name)) == digest,name


def payload(source, design, execution, slug):
    folder = execution/'payload'
    assert not folder.exists(), 'Do not overwrite a submitted input payload'
    folder.mkdir(parents=True)
    for name in ('qwen_calibration_encode_20261011.py','crossed_encode_20261011.py',
                 'qwen_external_encode_20261011.py','coderank_kaggle_20261010.py'):
        shutil.copy2(ROOT/name,folder/name)
    shutil.copy2(PROTOCOL,folder/PROTOCOL.name)
    shutil.copy2(source,folder/'selected_programs.jsonl')
    shutil.copy2(design,folder/'study_design.json')
    assert sha(MODEL_RECEIPT) == 'ef31e14d2f09874d637afee903a2632f555f90ba1a9682ee6e10b8c855282d5c'
    model = json.loads(MODEL_RECEIPT.read_text())
    assert model['settings'] == SETTINGS
    write(folder/'expected_model_files.json',model['model_files'])
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    receipt = {'machine_shape':'NvidiaTeslaT4','launcher_sha256':sha(execution/'kernel/run.py'),
        'design_sha256':sha(design),'settings':SETTINGS,
        'cohort':{'programs':len(rows),'roles':dict(Counter(r['role'] for r in rows)),
                  'source_sha256':sha(source)},
        'payload_sha256':{p.name:sha(p) for p in sorted(folder.iterdir())}}
    write(folder/'qwen_weight_execution.json',receipt)
    write(folder/'dataset-metadata.json',{'id':'ivankosyanenko/'+slug+'-inputs',
        'title':slug.replace('-',' '),'licenses':[{'name':'CC-BY-NC-4.0'}],
        'description':'Private research inference inputs. CodeContests subset; upstream source licenses apply. Code, source data and instructions only, no manuscript.'})
    write(execution/'kernel/kernel-metadata.json',{'id':'ivankosyanenko/'+slug,
        'title':slug.replace('-',' '),'code_file':'run.py','language':'python','kernel_type':'script',
        'is_private':True,'enable_gpu':True,'machine_shape':'NvidiaTeslaT4','enable_internet':True,
        'dataset_sources':['ivankosyanenko/'+slug+'-inputs'],'competition_sources':[],'kernel_sources':[]})
    return receipt


def prepare():
    if (OUT/'design_receipt.json').exists():
        design = json.loads((OUT/'design_receipt.json').read_text()); verify_bindings(design)
        return design
    rows,queries,gallery,indices,problems,folds = old.prepare()
    stage = json.loads((old.OUT/'ret_root.json').read_text())
    for name,digest in stage['output_sha256'].items(): assert sha(old.OUT/name)==digest
    selection = json.loads((old.OUT/'design_receipt.json').read_text())
    assert len(selection['query_ids'])==2600 and len(selection['gallery_ids'])==5200
    keep = set(selection['query_ids']+selection['gallery_ids'])
    sources = [json.loads(l) for l in (old.previous.COHORT/'selected_programs.jsonl').read_text().splitlines()]
    selected = [r for r in sources if r['source_relpath'] in keep]
    assert len(selected)==7800 and {r['source_relpath'] for r in selected}==keep
    assert {r['cluster_id'] for r in selected if r['role']=='query'}==set(problems)
    OUT.mkdir()
    with (OUT/'selected_programs.jsonl').open('x') as stream:
        for row in selected: stream.write(json.dumps(row,sort_keys=True)+'\n')
    paths = [PROTOCOL,Path(__file__),ROOT/'test_qwen_calibration_20261011.py',
        ROOT/'qwen_calibration_encode_20261011.py',ROOT/'crossed_encode_20261011.py',
        ROOT/'qwen_external_encode_20261011.py',ROOT/'coderank_kaggle_20261010.py',MODEL_RECEIPT,
        EXEC/'kernel/run.py',old.OUT/'design_receipt.json',old.OUT/'ret_root.json',
        old.OUT/'ret_root.npy',old.previous.COHORT/'selected_programs.jsonl',
        old.previous.DENSE/'D.npy',OUT/'selected_programs.jsonl',
        ROOT/'frequency_factorial_20261010.py',ROOT/'conditional_structure_pilot_20261010.py']
    fixed = {'created_before_new_scores_utc':datetime.now(timezone.utc).isoformat(),
        'bindings':{str(p):sha(p) for p in paths},'grid':GRID,'settings':SETTINGS,
        'problems':problems,'fold_assignment':selection['fold_assignment'],
        'query_ids':selection['query_ids'],'gallery_ids':selection['gallery_ids'],
        'old_query_indices':indices,'cutoff':100,'features_changed':False,
        'cpret_used_for_selection':False,'java_test_accessed':False}
    write(OUT/'design_receipt.json',fixed)
    shutil.copy2('/tmp/qwen-calibration-resources-20261011.json',OUT/'resources.json')
    payload(OUT/'selected_programs.jsonl',OUT/'design_receipt.json',EXEC,'code2hyp-qwen-calibration-20261011')
    return fixed


def encoding(execution, design):
    folder = execution/'output_v1/qwen_weight/encoding'
    inputs = execution/'payload'
    frozen = json.loads((inputs/'qwen_weight_execution.json').read_text())
    assert frozen['design_sha256']==sha(design)
    for name,digest in frozen['payload_sha256'].items(): assert sha(inputs/name)==digest,name
    saved = json.loads((folder/'encoding_receipt.json').read_text())
    assert saved['binding']==sha(inputs/'qwen_weight_execution.json')
    assert saved['settings']==SETTINGS and saved['gpu']=='Tesla T4'
    assert saved['source_sha256']==sha(inputs/'selected_programs.jsonl')
    assert saved['script_sha256']==sha(inputs/'crossed_encode_20261011.py')
    assert saved['model_files']==json.loads((inputs/'expected_model_files.json').read_text())
    for name in ('embeddings','token_lengths'): assert sha(folder/(name+'.npy'))==saved[name+'_sha256']
    vectors = np.load(folder/'embeddings.npy',allow_pickle=False)
    validate(vectors,frozen['cohort']['programs'])
    return vectors,saved


def evaluate():
    design = prepare()
    assert not (OUT/'result.json').exists(), 'Use check for a completed calibration'
    vectors,receipt = encoding(EXEC,OUT/'design_receipt.json')
    rows = [json.loads(l) for l in (OUT/'selected_programs.jsonl').read_text().splitlines()]
    by_id = {r['source_relpath']:i for i,r in enumerate(rows)}
    qi = [by_id[q] for q in design['query_ids']]; gi = [by_id[g] for g in design['gallery_ids']]
    labels = np.array([rows[i]['cluster_id'] for i in gi])
    qproblems = [rows[i]['cluster_id'] for i in qi]
    dense = vectors[qi]@vectors[gi].T
    structure = np.load(old.OUT/'ret_root.npy',allow_pickle=False,mmap_mode='r')
    assert structure.shape==dense.shape==(2600,5200)
    control = np.load(old.previous.DENSE/'D.npy',allow_pickle=False,mmap_mode='r')
    tie = np.array([s.encode() for s in design['gallery_ids']])
    values = np.empty((2600,len(GRID)))
    control_ap = []
    started = time.perf_counter()
    for i in range(len(qi)):
        d = dense[i].astype(float); s = structure[i]; positive = labels==qproblems[i]
        for j,alpha in enumerate(GRID): values[i,j]=metrics(rerank(d,s,alpha,tie),positive)[0]
        cd = -np.asarray(control[design['old_query_indices'][i]],dtype=float)
        control_ap.append([metrics(rerank(cd,s,a,tie),positive)[0] for a in (0.,.25)])
    problems = design['problems']
    masks = [np.array(qproblems)==p for p in problems]
    assert all(mask.sum()==8 for mask in masks)
    table = np.array([values[m].mean(axis=0) for m in masks])
    folds = np.array([design['fold_assignment'][p] for p in problems])
    choices,oof = old.choose_folds(table,folds)
    choice = select_weight(table)
    np.savez_compressed(OUT/'development.npz',query_grid=values,problem_grid=table,
        dense=dense,control_ap=np.array(control_ap),oof=oof)
    result = {'status':'development_only_not_external_confirmation','design_sha256':sha(OUT/'design_receipt.json'),
        'alpha':GRID[choice],'grid':GRID,'problem_means':table.mean(axis=0).tolist(),
        'dense_map8':float(table[:,0].mean()),'fixed_025_map8':float(table[:,GRID.index(.25)].mean()),
        'oof_map8':float(oof.mean()),'in_sample_selected_map8':float(table[:,choice].mean()),
        'folds':[{'fold':i,'alpha':GRID[c],'train_map8':float(table[folds!=i,c].mean()),
                  'held_map8':float(oof[folds==i].mean())} for i,c in enumerate(choices)],
        'coderank_dense_fixed025_map8':np.mean(control_ap,axis=0).tolist(),
        'encoding':receipt,'evaluation_seconds':time.perf_counter()-started,
        'arrays_sha256':sha(OUT/'development.npz'),'fresh_scores_seen':False,'cpret_scored':False}
    write(OUT/'result.json',result)
    check()
    write(OUT/'weight_seal.json',{'alpha':result['alpha'],'fixed_utc':datetime.now(timezone.utc).isoformat(),
        'result_sha256':sha(OUT/'result.json'),'verification_sha256':sha(OUT/'verification.json'),
        'design_sha256':sha(OUT/'design_receipt.json'),'primary_external':'selected_minus_dense_MAP8',
        'fresh_contents_read':False,'cpret_used':False})
    print(json.dumps({k:result[k] for k in ('alpha','dense_map8','fixed_025_map8','oof_map8','in_sample_selected_map8','folds')},indent=2))


def check():
    design = json.loads((OUT/'design_receipt.json').read_text()); verify_bindings(design)
    result = json.loads((OUT/'result.json').read_text())
    assert result['design_sha256']==sha(OUT/'design_receipt.json')
    assert result['arrays_sha256']==sha(OUT/'development.npz')
    rows = {r['source_relpath']:r for r in map(json.loads,(OUT/'selected_programs.jsonl').read_text().splitlines())}
    labels = np.array([rows[g]['cluster_id'] for g in design['gallery_ids']])
    tie = np.array([g.encode() for g in design['gallery_ids']])
    structure = np.load(old.OUT/'ret_root.npy',mmap_mode='r',allow_pickle=False)
    with np.load(OUT/'development.npz',allow_pickle=False) as data:
        table = data['problem_grid']; values=data['query_grid']; dense=data['dense']
        assert result['alpha']==GRID[select_weight(table)]
        folds=np.array([design['fold_assignment'][p] for p in design['problems']])
        choices,oof=old.choose_folds(table,folds)
        np.testing.assert_allclose(oof,data['oof'],rtol=0,atol=1e-12)
        for i,q in enumerate(design['query_ids']):
            relevant=np.flatnonzero(labels==rows[q]['cluster_id'])
            base=np.lexsort((tie,-dense[i]))
            fold=design['fold_assignment'][rows[q]['cluster_id']]
            for a in sorted({0.,.25,result['alpha'],GRID[choices[fold]]}):
                order=rerank(dense[i].astype(float),structure[i],a,tie)
                np.testing.assert_array_equal(order[100:],base[100:])
                ranks=np.empty(len(order),dtype=int);ranks[order]=np.arange(1,len(order)+1)
                rp=np.sort(ranks[relevant])
                ap=float(np.sum(np.arange(1,9)/rp*(rp<=8))/8)
                np.testing.assert_allclose(ap,values[i,GRID.index(a)],rtol=0,atol=1e-12)
        for i,p in enumerate(design['problems']):
            mask=np.array([rows[q]['cluster_id']==p for q in design['query_ids']])
            np.testing.assert_allclose(table[i],values[mask].mean(axis=0),rtol=0,atol=1e-12)
    seal={'status':'PASS','queries':2600,'problems':325,'result_sha256':sha(OUT/'result.json')}
    if (OUT/'verification.json').exists(): assert json.loads((OUT/'verification.json').read_text())==seal
    else: write(OUT/'verification.json',seal)
    print('PASS development-only IDs, hashes, independent AP8, selected weights and dense tails')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','evaluate','check'))
    mode=parser.parse_args().mode
    {'prepare':prepare,'evaluate':evaluate,'check':check}[mode]()
