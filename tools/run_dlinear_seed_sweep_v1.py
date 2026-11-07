#!/usr/bin/env python3
"""Controlled DLinear ETTh1 seed sensitivity on a pinned original source checkout.

Uses the original source files unchanged on disk. A fresh Python process per run
replaces ONLY `fix_seed = 2021` with the explicitly recorded run seed in memory.
Do not call the results unmodified original replication: they are sensitivity runs.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SHA='0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official',type=Path,default=Path.home()/'Documents/LTSF-Linear')
    p.add_argument('--project',type=Path,default=Path.home()/'Documents/COMP8240-LTSF-Linear-Reproduction')
    p.add_argument('--seeds',default='2021',help='Comma-separated integer seeds, e.g. 2021,2022,2023')
    p.add_argument('--horizons',default='96',help='Comma-separated ETTh1 horizons')
    p.add_argument('--seq-len',type=int,default=336)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--learning-rate',type=float,default=.005)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    official=a.official.resolve();project=a.project.resolve()
    seeds=[int(x) for x in a.seeds.split(',')]
    hs=[int(x) for x in a.horizons.split(',')]
    if not hs or not seeds or len(set(seeds))!=len(seeds) or len(set(hs))!=len(hs) or any(h not in (96,192,336,720) for h in hs):
        raise RuntimeError('Unexpected horizons or repeated seed')
    head=subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()
    if head!=SHA:raise RuntimeError('Unexpected official source HEAD: '+head)
    for extra in ([],['--cached']):
        if subprocess.call(['git','-C',str(official),'diff']+extra+['--quiet']):
            raise RuntimeError('Original checkout modified; stop')
    entry=official/'run_longExp.py'
    code=entry.read_text()
    if code.count('fix_seed = 2021')!=1:raise RuntimeError('Pinned entry script seed assignment changed')
    if not (official/'dataset/ETTh1.csv').exists():raise RuntimeError('ETTh1.csv not present at '+str(official/'dataset/ETTh1.csv'))
    outbase=project/'experiments/new_data/dlinear_seed_sweep_v1'
    sha=hashlib.sha256(code.encode()).hexdigest()
    runs=[(s,h,outbase/('h{}_seed{}'.format(h,s))) for h in hs for s in seeds]
    print('=== CONTROLLED DLINEAR SEED SWEEP ===')
    print('Original HEAD:',head,'| runner SHA256:',sha)
    print('Settings: seq_len',a.seq_len,'batch_size',a.batch_size,'LR',a.learning_rate)
    for seed,h,out in runs:
        print('PLAN horizon',h,'seed',seed,'dest',out,'EXISTS' if out.exists() else 'new')
    if a.dry_run:
        print('DLINEAR SEED SWEEP DRY RUN: PASS; no training or files written');return
    # subprocess runs a freshly compiled in-memory copy of the pinned entry
    # with only the explicit seed constant modified; official working tree stays clean.
    for seed,h,out in runs:
        if out.exists():raise RuntimeError('Existing sensitivity run preserved: '+str(out))
        out.mkdir(parents=True,exist_ok=False)
        meta=dict(source_sha=SHA,runner_sha256=sha,model='DLinear',dataset='ETTh1',
                  horizon=h,seed=seed,seq_len=a.seq_len,batch_size=a.batch_size,
                  learning_rate=a.learning_rate,seed_override='fix_seed constant only',
                  status='pending')
        (out/'config.json').write_text(json.dumps(meta,indent=2)+'\n')
        python_code='import pathlib,sys; p=pathlib.Path(sys.argv[1]); seed=sys.argv[2]; sys.path.insert(0,str(p.parent)); c=p.read_text(); assert c.count("fix_seed = 2021")==1; c=c.replace("fix_seed = 2021","fix_seed = "+seed); sys.argv=[str(p)]+sys.argv[3:]; exec(compile(c,str(p),"exec"),{"__name__":"__main__","__file__":str(p)})'
        argv=[sys.executable,'-u','-c',python_code,str(entry),str(seed),
              '--is_training','1','--root_path',str(official/'dataset'),
              '--data_path','ETTh1.csv','--model_id','ETTh1_{}_{}_seed{}'.format(a.seq_len,h,seed),
              '--model','DLinear','--data','ETTh1','--features','M',
              '--seq_len',str(a.seq_len),'--label_len','48','--pred_len',str(h),
              '--enc_in','7','--dec_in','7','--c_out','7',
              '--des','COMP8240_SeedSweep','--itr','1','--batch_size',str(a.batch_size),
              '--learning_rate',str(a.learning_rate),'--train_epochs','10',
              '--patience','3','--num_workers','0','--checkpoints',str(out/'checkpoints')]
        print('RUN h',h,'seed',seed,'log',out/'run.log',flush=True)
        with (out/'run.log').open('w') as log:
            proc=subprocess.run(argv,cwd=out,stdout=log,stderr=subprocess.STDOUT)
        meta['status']='PASS' if proc.returncode==0 else 'FAILED'
        meta['return_code']=proc.returncode
        (out/'config.json').write_text(json.dumps(meta,indent=2)+'\n')
        if proc.returncode:raise RuntimeError('Seed run failed; inspect '+str(out/'run.log'))
        print('PASS h',h,'seed',seed,flush=True)
    print('DLINEAR SEED SENSITIVITY: COMPLETE')


if __name__=='__main__':main()
