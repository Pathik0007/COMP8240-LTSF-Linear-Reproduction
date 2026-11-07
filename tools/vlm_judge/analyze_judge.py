#!/usr/bin/env python3
"""Compute blinded visual-judge reliability against held-out numerical scores."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def score_analysis(manifest, answers):
    items = {o['id']:o for o in manifest}
    outputs = {o['id']:o for o in answers}
    if len(outputs)!=len(answers):
        raise RuntimeError('Duplicate judge output IDs')
    bybase = defaultdict(dict)
    rows=[]
    for ident, case in items.items():
        out=outputs.get(ident)
        if not out:
            continue
        choice=out.get('parsed', {}).get('choice') if out.get('valid') else None
        row={'id':ident,'base_id':case['base_id'],'condition':case['condition'],
             'family':case['family'],'variant':case['variant'],'judge_choice':choice or 'INVALID',
             'gold_mae':case['gold_mae'],'gold_mse':case['gold_mse'],
             'mae_agrees':choice==case['gold_mae'] if choice else False,
             'mse_agrees':choice==case['gold_mse'] if choice else False,
             'valid':bool(out.get('valid')),
             'A_mae':case['A_mae'],'B_mae':case['B_mae'],
             'A_mse':case['A_mse'],'B_mse':case['B_mse']}
        rows.append(row)
        bybase[case['base_id']][case['variant']]=row
    original=[r for r in rows if r['variant']=='ab']
    reversed_consistency=[]
    for base, variants in bybase.items():
        if {'ab','ba'} <= variants.keys():
            a=variants['ab']['judge_choice'];b=variants['ba']['judge_choice']
            if 'INVALID' not in (a,b):
                flip={'A':'B','B':'A','tie':'tie','abstain':'abstain'}
                reversed_consistency.append(flip[a]==b)
    valid=[r for r in original if r['valid']]
    definitive=[r for r in valid if r['gold_mae'] in ('A','B') and r['judge_choice'] in ('A','B')]
    result={
        'base_cases_in_manifest':len({c['base_id'] for c in manifest}),
        'all_panels_in_manifest':len(manifest),
        'judge_outputs_received':len(answers),
        'original_cases_received':len(original),
        'original_valid_json':len(valid),
        'invalid_original':len(original)-len(valid),
        'original_decisions':dict(Counter(r['judge_choice'] for r in original)),
        'original_mae_agreement_including_ties':sum(r['mae_agrees'] for r in valid)/len(valid) if valid else None,
        'original_mse_agreement_including_ties':sum(r['mse_agrees'] for r in valid)/len(valid) if valid else None,
        'definitive_mae_eligible_cases':len(definitive),
        'definitive_mae_agreement':sum(r['mae_agrees'] for r in definitive)/len(definitive) if definitive else None,
        'position_consistency_n':len(reversed_consistency),
        'position_consistency_rate':sum(reversed_consistency)/len(reversed_consistency) if reversed_consistency else None,
        'by_condition':{},
    }
    for c in sorted({r['condition'] for r in original}):
        group=[r for r in original if r['condition']==c and r['valid']]
        result['by_condition'][c]={
            'valid_original':len(group),
            'mae_agreement_rate':sum(r['mae_agrees'] for r in group)/len(group) if group else None,
            'mse_agreement_rate':sum(r['mse_agrees'] for r in group)/len(group) if group else None,
        }
    return rows, result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--answers',type=Path,required=True)
    p.add_argument('--out-prefix',type=Path,required=True)
    a=p.parse_args()
    manifest=[json.loads(x) for x in a.cases.read_text().splitlines() if x.strip()]
    answers=[json.loads(x) for x in a.answers.read_text().splitlines() if x.strip()]
    rows, report=score_analysis(manifest,answers)
    a.out_prefix.parent.mkdir(parents=True,exist_ok=True)
    (a.out_prefix.with_suffix('.json')).write_text(json.dumps(report,indent=2))
    with a.out_prefix.with_suffix('.csv').open('w',newline='') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('=== FORECAST JUDGE RELIABILITY ===')
    print(json.dumps(report,indent=2))
    print('Note: these are descriptive rates for selected/overlapping windows, not independent-trial p-values.')

if __name__=='__main__':
    main()
