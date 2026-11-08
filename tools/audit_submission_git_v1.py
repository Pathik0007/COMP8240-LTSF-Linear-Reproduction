#!/usr/bin/env python3
"""Conservative submission staging: DRY RUN by default, explicitly --stage to git add.
Does not push/commit. Fails on oversized files, secrets, or staged unexpected paths.
"""
import argparse,re,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ALLOWED_EXACT={
  'experiments/original_etth1/run_linear_nlinear_etth1.sh',
  'environment/environment_notes.md',
  'environment/environment.yml',
  'data/README.md',
  'experiments/new_data/vlm_judge_v3/claude46_judgments_errors.jsonl',
  '.gitignore','README.md','tools/vlm_judge/README.md',
  'docs/EXPERIMENTS.md','docs/REPRODUCE.md',
  'results/original_replication.csv','results/comparison_table.csv',
  'experiments/new_data/vlm_judge_v3/public_cases.jsonl',
  'experiments/new_data/vlm_judge_v3/cases.jsonl',
  'experiments/new_data/vlm_judge_v3/claude46_judgments.jsonl',
  'experiments/new_data/vlm_judge_v3/human_sample_v1.jsonl',
  'experiments/new_data/vlm_judge_v3/human_annotations_rater1.csv',
  'experiments/new_data/vlm_judge_v3/human_annotations_rater1_batch2.csv'
}
PATTERNS=[
  'src/**/*.py','src/**/*.sh','tools/**/*.py','tests/test_verified_fixes_v1.py',
  'metadata/*.json','metadata/*.csv',
  'results/new_data/*.csv','results/new_data/*.json',
  'results/figures/**/*.png',
  'experiments/new_data/dlinear_seed_sweep_v1/h*_seed*/config.json',
  'experiments/new_data/dlinear_seed_sweep_v1/h*_seed*/run.log',
  'experiments/new_data/pgcb_valid_windows_v1/protocol.json',
  'experiments/new_data/pgcb_valid_windows_v2/protocol.json',
  'experiments/new_data/vlm_judge_v3/images/*.png'
]
PRIVATE_WORDS=('secret','password','token','api_key')
KEY_RE=[re.compile(rb'sk-ant-[A-Za-z0-9_-]{20,}'),re.compile(rb'sk-proj-[A-Za-z0-9_-]{20,}'),
        re.compile(rb'ghp_[A-Za-z0-9]{30,}'),re.compile(rb'github_pat_[A-Za-z0-9_]{25,}')]
MAX_ONE=25_000_000
MAX_TOTAL=95_000_000

def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()

def main():
    q=argparse.ArgumentParser()
    q.add_argument('--stage',action='store_true',help='Actually git add audited files')
    args=q.parse_args()
    if git('rev-parse','--show-toplevel')!=str(ROOT):
        raise RuntimeError('Script must run in expected root git repo')
    files=set(ALLOWED_EXACT)
    for pattern in PATTERNS:
        files|={str(p.relative_to(ROOT)) for p in ROOT.glob(pattern) if p.is_file()}
    files={p for p in files if (ROOT/p).is_file()}
    if not files:
        raise RuntimeError('No files available')
    total=0
    for name in sorted(files):
        p=ROOT/name
        if p.is_symlink() or p.stat().st_size>MAX_ONE:
            raise RuntimeError('Unexpected symlink / oversized file: '+name)
        if any(k in name.lower() for k in ('private_key','billing','credentials')):
            raise RuntimeError('Private filename: '+name)
        raw=p.read_bytes()
        total+=len(raw)
        for rx in KEY_RE:
            if rx.search(raw):
                raise RuntimeError('Possible credential in '+name)
        if '/images/' not in name and (b'-----BEGIN' + b' PRIVATE KEY-----') in raw:
            raise RuntimeError('Private key material in '+name)
    if total>MAX_TOTAL:
        raise RuntimeError('Excessive total staging bytes: '+str(total))
    staged={x for x in git('diff','--cached','--name-only').splitlines() if x}
    unknown=staged-files
    if unknown:
        raise RuntimeError('Unapproved files already staged: '+repr(sorted(unknown)))
    print('=== CURATED GIT CHECKPOINT ===')
    print('Candidate files:',len(files),'| Total MB:',round(total/1e6,2))
    for f in sorted(files):
        print(('STAGED' if f in staged else 'CANDIDATE'),f)
    if not args.stage:
        print('GIT AUDIT DRY RUN: PASS; NO STAGING OR COMMIT')
        return
    for f in sorted(files):
        subprocess.run(['git','-C',str(ROOT),'add','--',f],check=True)
    subprocess.run(['git','-C',str(ROOT),'diff','--cached','--check'],check=True)
    print('GIT CURATED STAGING: PASS. REVIEW git diff --cached --stat BEFORE COMMIT.')

if __name__=='__main__':
    main()
