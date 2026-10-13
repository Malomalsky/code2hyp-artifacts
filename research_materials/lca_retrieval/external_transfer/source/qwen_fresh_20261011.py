"""Evaluate a sealed development weight once on unused CodeContests problems."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.sparse import load_npz
import qwen_calibration_20261011 as development
import top100_fresh_20261010 as old
from rank_fusion_followup import ROOT, sha, write
from cpret_evaluate import query_metrics, KEYS

OUT=ROOT/'followup_results/qwen_fresh_20261011'
EXEC=development.EXEC.parent/'kaggle_qwen_fresh_20261011'
SHARDS=(
    ('valid-00000-of-00001-5e672c5751f060d3.parquet',51829044,'02e8c1ccedae716f1e43cc813fcb7823c3db666ea92638820aba80e8cef451ab'),
    ('test-00000-of-00001-9c49eeff30aacaa8.parquet',63077400,'aa426cbdb202bf8703b658bcb31fd1878ca7cfd33ca07d3b703dc94ca6a2b651'))
METHODS=('dense','fixed025','selected')


def weight():
    development.check()
    seal=json.loads((development.OUT/'weight_seal.json').read_text())
    assert seal['result_sha256']==sha(development.OUT/'result.json')
    assert seal['verification_sha256']==sha(development.OUT/'verification.json')
    assert seal['design_sha256']==sha(development.OUT/'design_receipt.json')
    assert seal['alpha']==json.loads((development.OUT/'result.json').read_text())['alpha']
    assert not seal['fresh_contents_read'] and not seal['cpret_used']
    return seal['alpha']


def exclusions():
    ids,hashes,bindings=old.exclusions()
    seal=json.loads((old.OUT/'selection_seal.json').read_text())
    for name,digest in seal['sha256'].items():
        assert sha(old.OUT/name)==digest,name
        bindings[str(old.OUT/name)]=digest
    for row in map(json.loads,(old.OUT/'metadata.jsonl').read_text().splitlines()):
        ids.add(row['cluster_id'])
        for h,k in zip(hashes,('d0','d1','d2')): h.add(row[k])
    for name,size,digest in old.SHARDS:
        path=old.OUT/'input'/name
        assert path.stat().st_size==size and sha(path)==digest
        bindings[str(path)]=digest
        for batch in old.previous.cpret_prepare.pq.ParquetFile(path).iter_batches(
                columns=['source','cf_contest_id','cf_index'],use_threads=False):
            ids.update(f"{r['cf_contest_id']}:{r['cf_index']}" for r in batch.to_pylist() if r['source']==2)
    return ids,hashes,bindings


def prepare():
    alpha=weight()
    if (OUT/'design_receipt.json').exists():
        design=json.loads((OUT/'design_receipt.json').read_text())
        development.verify_bindings(design)
        assert design['alpha']==alpha
        return design
    _,_,bindings=exclusions()
    inherited=json.loads((old.OUT/'design_receipt.json').read_text())['bindings']
    for path,digest in inherited.items(): assert sha(Path(path))==digest,path
    bindings.update(inherited)
    paths=[Path(__file__),ROOT/'test_qwen_fresh_20261011.py',development.PROTOCOL,
        ROOT/'qwen_calibration_20261011.py',development.OUT/'weight_seal.json',
        development.OUT/'result.json',development.OUT/'verification.json']
    bindings.update({str(p):sha(p) for p in paths})
    OUT.mkdir()
    design={'created_before_new_contents_utc':datetime.now(timezone.utc).isoformat(),
        'bindings':bindings,'alpha':alpha,'shards':SHARDS,'min_problems':30,
        'all_eligible':True,'bootstrap_seed':20261014,'bootstrap_resamples':20000,
        'primary':'selected_minus_dense_MAP8','java_test_accessed':False,
        'full_historical_fingerprint_inventory_available':False}
    write(OUT/'design_receipt.json',design)
    return design


def select():
    prepare()
    if (OUT/'selection_seal.json').exists():
        for name,digest in json.loads((OUT/'selection_seal.json').read_text())['sha256'].items():
            assert sha(OUT/name)==digest,name
        return [json.loads(l) for l in (OUT/'selected_programs.jsonl').read_text().splitlines()]
    assert not (OUT/'selection_receipt.json').exists(),'Preserve earlier selection/shortfall'
    ids,hashes,_=exclusions()
    old.previous.download(SHARDS,OUT/'input')
    candidates=[];attrition=Counter()
    for name,_,_ in SHARDS:
        for record in old.previous.old.previous.records(OUT/'input'/name):
            attrition['input_problems']+=1
            if record['source']!=2:
                attrition['non_codeforces']+=1;continue
            problem=f"{record['cf_contest_id']}:{record['cf_index']}"
            if problem in ids:
                attrition['previous_problem']+=1;continue
            for language,source in zip(record['solutions']['language'],record['solutions']['solution']):
                if language!=3: continue
                row,reason=old.previous.old.previous.candidate(source)
                attrition['python3_candidates']+=1
                if reason: attrition[reason]+=1
                elif any(row[k] in h for k,h in zip(('d0','d1','d2'),hashes)):
                    attrition['previous_fingerprint']+=1
                else: candidates.append({**row,'cluster_id':problem,'name':record['name']})
        print('Eligibility processed',name,flush=True)
    _,eligible,removed=old.previous.choose(candidates,1)
    attrition.update(removed)
    write(OUT/'selection_receipt.json',{'eligible_problems':eligible,'attrition':dict(attrition),
        'excluded_problem_ids':sorted(ids),'excluded_fingerprint_counts':list(map(len,hashes)),
        'retrieval_scores_seen':False,'weight_seal_sha256':sha(development.OUT/'weight_seal.json')})
    if eligible<30:
        write(OUT/'status.json',{'status':'insufficient_eligible_problems_no_scores','problems':eligible})
        return []
    rows,count,_=old.previous.choose(candidates,eligible)
    assert count==eligible and len(rows)==16*eligible
    assert not {r['cluster_id'] for r in rows}&ids
    for k,h in zip(('d0','d1','d2'),hashes):
        assert len({r[k] for r in rows})==len(rows) and not {r[k] for r in rows}&h
    for name,source in (('selected_programs.jsonl',True),('metadata.jsonl',False)):
        with (OUT/name).open('x') as stream:
            for row in rows: stream.write(json.dumps({k:v for k,v in row.items() if source or k!='source'},sort_keys=True)+'\n')
    write(OUT/'selection_seal.json',{'fixed_before_scores_utc':datetime.now(timezone.utc).isoformat(),
        'sha256':{n:sha(OUT/n) for n in ('selected_programs.jsonl','metadata.jsonl','selection_receipt.json','design_receipt.json')}})
    print('SEALED',eligible,'new problems;',len(rows),'programs',flush=True)
    return rows


def prepare_run():
    rows=select()
    if not rows: return
    original=old.previous.OUT
    try:
        old.previous.OUT=OUT
        old.previous.features(rows)
    finally: old.previous.OUT=original
    if (EXEC/'payload').exists():
        receipt=json.loads((EXEC/'payload/qwen_weight_execution.json').read_text())
        assert receipt['design_sha256']==sha(OUT/'design_receipt.json')
        for name,digest in receipt['payload_sha256'].items(): assert sha(EXEC/'payload'/name)==digest
        return
    (EXEC/'kernel').mkdir(parents=True)
    shutil.copy2(development.EXEC/'kernel/run.py',EXEC/'kernel/run.py')
    development.payload(OUT/'selected_programs.jsonl',OUT/'design_receipt.json',EXEC,'code2hyp-qwen-fresh-20261011')


def contrast(delta,resamples):
    values=delta[resamples].mean(axis=1)
    return {'mean':float(delta.mean()),'percentile95':np.quantile(values,[.025,.975]).tolist(),
        'higher':int((delta>1e-12).sum()),'lower':int((delta < -1e-12).sum()),
        'equal':int((np.abs(delta)<=1e-12).sum())}


def evaluate():
    design=prepare();assert not (OUT/'result.json').exists(),'Use check after completion'
    rows=select();assert rows
    for name,digest in json.loads((OUT/'features.json').read_text())['sha256'].items(): assert sha(OUT/name)==digest,name
    vectors,receipt=development.encoding(EXEC,OUT/'design_receipt.json')
    qi=np.array([i for i,r in enumerate(rows) if r['role']=='query'])
    gi=np.array([i for i,r in enumerate(rows) if r['role']=='gallery'])
    dense=vectors[qi]@vectors[gi].T
    features=load_npz(OUT/'ret_root_features.npz')
    structural=(features[qi]@features[gi].T).toarray()
    tie=np.array([rows[i]['source_relpath'].encode() for i in gi])
    labels=np.array([rows[i]['cluster_id'] for i in gi])
    groups=np.array([rows[i]['cluster_id'] for i in qi]);problems=np.unique(groups)
    quality=np.empty((3,len(qi),4));rankings=np.empty((3,len(qi),len(gi)),dtype=np.int32)
    started=time.perf_counter()
    for i,q in enumerate(qi):
        for j,a in enumerate((0.,.25,design['alpha'])):
            order=development.rerank(dense[i].astype(float),structural[i],a,tie)
            rankings[j,i]=order
            quality[j,i]=development.metrics(order,labels==rows[q]['cluster_id'])
    means=np.stack([quality[:,groups==p].mean(axis=1) for p in problems],axis=1)
    rng=np.random.Generator(np.random.PCG64(20261014))
    resamples=rng.integers(0,len(problems),size=(20000,len(problems)))
    contrasts={name:contrast(means[2,:,0]-means[j,:,0],resamples)
        for j,name in ((0,'selected_minus_dense'),(1,'selected_minus_fixed025'))}
    np.savez_compressed(OUT/'arrays.npz',rankings=rankings,quality=quality,problem_means=means,
        dense=dense,structure=structural)
    write(OUT/'result.json',{'status':'completed_fixed_weight_new_problems','alpha':design['alpha'],
        'queries':len(qi),'gallery':len(gi),'problems':problems.tolist(),
        'methods':{name:dict(zip(('MAP@8','nDCG@10','MRR@10','Recall@100'),means[j].mean(axis=0).tolist()))
            for j,name in enumerate(METHODS)},'contrasts':contrasts,
        'design_sha256':sha(OUT/'design_receipt.json'),'selection_sha256':sha(OUT/'selection_seal.json'),
        'weight_seal_sha256':sha(development.OUT/'weight_seal.json'),
        'features_sha256':sha(OUT/'features.json'),'arrays_sha256':sha(OUT/'arrays.npz'),
        'encoding':receipt,'scoring_seconds':time.perf_counter()-started,
        'parameters_fit_on_new_cohort':False,'java_test_accessed':False})
    check()
    print(json.dumps(json.loads((OUT/'result.json').read_text()),indent=2))


def check():
    design=json.loads((OUT/'design_receipt.json').read_text());development.verify_bindings(design)
    result=json.loads((OUT/'result.json').read_text())
    for key,name in (('design_sha256','design_receipt.json'),('selection_sha256','selection_seal.json'),
                     ('features_sha256','features.json'),('arrays_sha256','arrays.npz')):
        assert sha(OUT/name)==result[key]
    rows=[json.loads(l) for l in (OUT/'metadata.jsonl').read_text().splitlines()]
    queries=[r for r in rows if r['role']=='query'];gallery=[r for r in rows if r['role']=='gallery']
    ids=[r['source_relpath'] for r in gallery];tie=np.array([s.encode() for s in ids])
    with np.load(OUT/'arrays.npz',allow_pickle=False) as saved:
        actual=np.empty_like(saved['quality'])
        for i,q in enumerate(queries):
            rel={g['source_relpath']:1 for g in gallery if g['cluster_id']==q['cluster_id']}
            base=np.lexsort((tie,-saved['dense'][i]))
            for j,a in enumerate((0.,.25,design['alpha'])):
                mixed=(1-a)*saved['dense'][i,base[:100]].astype(float)+a*saved['structure'][i,base[:100]]
                head=sorted(range(len(mixed)),key=lambda k:(-mixed[k],tie[base[k]]))
                order=np.concatenate((base[np.array(head)],base[100:]))
                np.testing.assert_array_equal(order,saved['rankings'][j,i])
                common=query_metrics(order,ids,rel)
                pos=np.array([k+1 for k,g in enumerate(order) if ids[g] in rel]);top=pos[pos<=8]
                ap=float(np.sum(np.arange(1,len(top)+1)/top)/8)
                actual[j,i]=[ap,*[common[k] for k in KEYS]]
        np.testing.assert_allclose(actual,saved['quality'],rtol=0,atol=1e-12)
        groups=np.array([q['cluster_id'] for q in queries]);problems=np.unique(groups)
        means=np.stack([actual[:,groups==p].mean(axis=1) for p in problems],axis=1)
        np.testing.assert_allclose(means,saved['problem_means'],rtol=0,atol=1e-12)
        rng=np.random.Generator(np.random.PCG64(20261014));samples=rng.integers(0,len(problems),size=(20000,len(problems)))
        for j,name in ((0,'selected_minus_dense'),(1,'selected_minus_fixed025')):
            assert contrast(means[2,:,0]-means[j,:,0],samples)==result['contrasts'][name]
    seal={'status':'PASS','result_sha256':sha(OUT/'result.json'),'queries':len(queries),'methods':3}
    if (OUT/'verification.json').exists(): assert json.loads((OUT/'verification.json').read_text())==seal
    else: write(OUT/'verification.json',seal)
    print('PASS independent ranks, AP8/nDCG/MRR/recall, problem means and paired contrasts')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','evaluate','check'))
    {'prepare':prepare_run,'evaluate':evaluate,'check':check}[p.parse_args().mode]()
