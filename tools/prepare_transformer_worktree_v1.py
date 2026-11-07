#!/usr/bin/env python3
"""Prepare isolated, documented non-Linear CPU forward-argument fix.

The original pinned LTSF-Linear directory stays untouched. This generates
an explicitly patched *detached git worktree* for optional Informer/Autoformer
exploratory baselines. No model training is performed by this command.
"""
import argparse
import hashlib
import subprocess
from pathlib import Path

SHA='0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6'
BAD='outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)'
GOOD='outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)'


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--official',type=Path,default=Path.home()/'Documents/LTSF-Linear')
    a.add_argument('--worktree',type=Path,default=Path.home()/'Documents/LTSF-Linear-TransformerFix')
    a.add_argument('--dry-run',action='store_true')
    args=a.parse_args()
    official=args.official.resolve();worktree=args.worktree.expanduser().resolve()
    head=subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()
    if head!=SHA:raise RuntimeError('Source HEAD differs: '+head)
    src=official/'exp/exp_main.py'
    code=src.read_text()
    if code.count(BAD)!=1:raise RuntimeError('The pinned CPU call did not match exactly; stop')
    if worktree.exists():raise RuntimeError('Existing worktree path: '+str(worktree)+'; preserving; stop')
    print('Original pinned HEAD:',head)
    print('Affected file:',src)
    print('Exact bad-call occurrences:',code.count(BAD))
    print('Separate patched worktree will be:',worktree)
    if args.dry_run:
        print('TRANSFORMER FIX DRY RUN: PASS; no source touched');return
    subprocess.run(['git','-C',str(official),'worktree','add','--detach',str(worktree),SHA],check=True)
    path=worktree/'exp/exp_main.py'
    original=path.read_text()
    if original.count(BAD)!=1:raise RuntimeError('Unexpected worktree version; investigate')
    corrected=original.replace(BAD,GOOD)
    compile(corrected,str(path),'exec')
    path.write_text(corrected)
    doc=(worktree/'COMP8240_CPU_TRANSFORMER_PATCH.md')
    doc.write_text('# COMP8240 CPU Transformer fix\n\n'
      'Pinned upstream commit: '+SHA+'\n\n'
      'File: exp/exp_main.py (ordinary, non-AMP CPU training branch).\n'
      'Replace one five-argument `self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)` call '\
      'with the four-argument `self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)` call.\n'
      'Original SHA256: '+hashlib.sha256(original.encode()).hexdigest()+'\n'
      'Patched SHA256: '+hashlib.sha256(corrected.encode()).hexdigest()+'\n'
      'This is a documented deviation from the original repository, not an unmodified replication.\n')
    print('ISOLATED TRANSFORMER WORKTREE: PASS')
    print('Patch diff: git -C',worktree,'diff -- exp/exp_main.py')


if __name__=='__main__':main()
