#!/usr/bin/env python3
"""Finish visually blind human labels; no gold labels or Claude case judgments shown.

This is a post-hoc annotation round performed after aggregate judge results
were viewed; it is not an independent pre-registered or multi-rater study.
"""
import argparse
import csv
import json
import random
import subprocess
from datetime import datetime,timezone
from pathlib import Path


def ids(path):
    if not path.exists():return set()
    with path.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if not rows:return set()
    if 'id' not in rows[0]:raise RuntimeError('Missing id in '+str(path))
    out=[r['id'] for r in rows]
    if len(out)!=len(set(out)):raise RuntimeError('Duplicate annotation ID in '+str(path))
    return set(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('experiments/new_data/vlm_judge_v3'))
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    root=a.root.resolve()
    first=ids(root/'human_annotations_rater1.csv')
    out=root/'human_annotations_rater1_batch2.csv'
    done=ids(out)
    cases=[json.loads(x) for x in (root/'public_cases.jsonl').read_text().splitlines() if x.strip()]
    if len(cases)!=144 or len({c['id'] for c in cases})!=144:
        raise RuntimeError('Unexpected public blind manifest')
    ab=[c for c in cases if c['variant']=='ab']
    if len(ab)!=72:raise RuntimeError('Expected 72 original-orientation panels')
    needed=[c for c in ab if c['id'] not in first]
    if len(first)!=24 or len(needed)!=48:raise RuntimeError('Expected prior 24 labels plus 48 remaining')
    if not done <= {c['id'] for c in needed}:raise RuntimeError('Unexpected batch2 case ID')
    random.Random(8240).shuffle(needed)
    print('HUMAN LABEL AUDIT: original',len(first),'remaining',len(needed),'batch2 already',len(done),flush=True)
    for case in needed:
        if not (root/case['image']).is_file():raise RuntimeError('Missing image: '+case['image'])
    if a.dry_run:
        print('DRY RUN: PASS; NO IMAGES OPENED AND NO ANNOTATIONS WRITTEN');return
    header=['id','choice','confidence','phase','annotated_at_utc']
    if out.exists() and out.stat().st_size:
        with out.open(newline='',encoding='utf-8') as f:
            if list(csv.reader(f).__next__())!=header:
                raise RuntimeError('Existing batch2 CSV has unexpected header')
    with out.open('a',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        if not out.stat().st_size:w.writerow(header);f.flush()
        for i,case in enumerate(needed,1):
            if case['id'] in done:continue
            subprocess.run(['open',str(root/case['image'])],check=True)
            ans=''
            while ans not in ('a','b','tie','q'):
                ans=input('[{}/{}] A, B or tie (q=stop): '.format(i,len(needed))).strip().lower()
            if ans=='q':break
            conf=''
            while conf not in ('1','2','3','4','5'):
                conf=input('Confidence 1-5: ').strip()
            w.writerow([case['id'],ans.upper() if ans in ('a','b') else ans,int(conf),'posthoc_after_aggregate_judge_review',datetime.now(timezone.utc).isoformat()]);f.flush()
            done.add(case['id'])
    print('HUMAN BATCH2 SAVED:',len(done),'/',len(needed),out)


if __name__=='__main__':main()
