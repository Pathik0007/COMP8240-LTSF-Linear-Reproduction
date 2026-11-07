#!/usr/bin/env python3
"""COMP8240: auditable, offline, pair-aware VLM forecast-judge evaluation.

Only saved blind-case manifests, numerical gold labels, LLM results and optional
human annotations are read. No API access, retraining, or alterations to input.

The two earliest human sample cases are treated as AI-assisted per recorded
conversation; excluded from 'independent-human' summaries (not deleted).
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

LABELS = ('A', 'B', 'tie', 'abstain')
SWAP = {'A': 'B', 'B': 'A', 'tie': 'tie', 'abstain': 'abstain'}


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def require_unique(rows, key, label):
    keys = [r[key] for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError('Duplicate ' + key + ' in ' + label)


def normalise(value):
    s = str(value).strip().lower()
    if s == 'a': return 'A'
    if s == 'b': return 'B'
    if s in ('tie', 'abstain'): return s
    raise ValueError('Invalid choice: ' + repr(value))


def ratio(successes, total):
    return round(successes / total, 6) if total else None


def counts_choices(rows, key):
    return dict(sorted(Counter(r[key] for r in rows).items()))


def agreement(a, b):
    assert len(a) == len(b)
    return {'matches': sum(x == y for x, y in zip(a, b)),
            'n': len(a),
            'rate': ratio(sum(x == y for x, y in zip(a, b)), len(a))}


def kappa(a, b):
    if len(a) != len(b): raise RuntimeError('Mismatched rater vectors')
    if not a: return None
    labels = sorted(set(a + b))
    pa = Counter(a)
    pb = Counter(b)
    po = sum(x == y for x, y in zip(a, b)) / len(a)
    pe = sum((pa[c]/len(a))*(pb[c]/len(a)) for c in labels)
    return round((po-pe)/(1-pe), 6) if pe < 1 else None


def confusion(gold, predicted):
    table = {g: {j: 0 for j in LABELS} for g in LABELS}
    for x, y in zip(gold, predicted):
        table[x][y] += 1
    return {g: vals for g, vals in table.items() if sum(vals.values())}


def metric_winner(a, b, threshold=0.05):
    a, b = float(a), float(b)
    if not (a >= 0 and b >= 0):
        raise RuntimeError('Negative MAE/MSE cannot be a valid reference score')
    if abs(a - b) <= threshold * max(a, b, 1e-12):
        return 'tie'
    return 'A' if a < b else 'B'


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    names = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--answers', type=Path, required=True)
    p.add_argument('--human', type=Path)
    p.add_argument('--sample', type=Path)
    p.add_argument('--out-prefix', type=Path, required=True)
    a = p.parse_args()

    cases = read_jsonl(a.cases)
    answers = read_jsonl(a.answers)
    require_unique(cases, 'id', 'private manifest')
    require_unique(answers, 'id', 'judge responses')
    if len(cases) != 144 or len(answers) != 144:
        raise RuntimeError('Expected 144 private cases and 144 judge outputs')
    by_id = {r['id']: r for r in cases}
    by_ans = {r['id']: r for r in answers}
    if set(by_id) != set(by_ans):
        raise RuntimeError('Judge responses do not match private case manifest')
    if any(not r.get('valid') for r in answers):
        raise RuntimeError('Not all judge outputs are valid')
    served = sorted({str(r.get('served_model')) for r in answers})
    if len(served) != 1 or served[0] != 'claude-sonnet-4-6':
        raise RuntimeError('Unexpected served model: '+repr(served))
    prompts = sorted({str(r.get('prompt_sha256')) for r in answers})
    if len(prompts) != 1:
        raise RuntimeError('Multiple prompt hashes; investigate')
    versions = Counter(str(r.get('max_tokens')) for r in answers)

    base_groups = defaultdict(dict)
    panels = []
    for r in cases:
        q = by_ans[r['id']]
        chosen = normalise(q['parsed']['choice'])
        if q.get('image_sha256') != r.get('image_sha256'):
            raise RuntimeError('Judge image SHA differs from private manifest: '+r['id'])
        if abs(float(r.get('gold_tie_threshold', .05)) - .05) > 1e-12:
            raise RuntimeError('Unexpected gold-label tie threshold')
        if normalise(r['gold_mae']) != metric_winner(r['A_mae'], r['B_mae']):
            raise RuntimeError('MAE gold label is not supported by saved numerical values: '+r['id'])
        if normalise(r['gold_mse']) != metric_winner(r['A_mse'], r['B_mse']):
            raise RuntimeError('MSE gold label is not supported by saved numerical values: '+r['id'])
        if r['variant'] not in ('ab', 'ba'):
            raise RuntimeError('Unknown A/B variant')
        gold_mae = normalise(r['gold_mae'])
        gold_mse = normalise(r['gold_mse'])
        panel = {
            'id': r['id'], 'base_id': r['base_id'],
            'family': r['family'], 'condition': r['condition'],
            'variant': r['variant'], 'choice': chosen,
            'gold_mae': gold_mae, 'gold_mse': gold_mse,
            'mae_match': int(chosen == gold_mae),
            'mse_match': int(chosen == gold_mse),
            'confidence': q['parsed']['confidence'],
            'reason': q['parsed']['reason'],
            'image': r['image'],
            'A_mae': r['A_mae'], 'B_mae': r['B_mae'],
            'A_mse': r['A_mse'], 'B_mse': r['B_mse'],
        }
        if r['variant'] in base_groups[r['base_id']]:
            raise RuntimeError('Duplicate pair variant: ' + r['base_id'])
        base_groups[r['base_id']][r['variant']] = panel
        panels.append(panel)
    if len(base_groups) != 72 or any(set(g) != {'ab', 'ba'} for g in base_groups.values()):
        raise RuntimeError('Expected exactly 72 complete AB/BA pairs')

    orig = [base_groups[k]['ab'] for k in sorted(base_groups)]
    reversed_rows = [base_groups[k]['ba'] for k in sorted(base_groups)]
    pairs = []
    for r in orig:
        b = base_groups[r['base_id']]['ba']
        if r['gold_mae'] != SWAP[b['gold_mae']] or r['gold_mse'] != SWAP[b['gold_mse']]:
            raise RuntimeError('Gold labels fail reversal validation: ' + r['base_id'])
        b_normalized = SWAP[b['choice']]
        consistent = r['choice'] == b_normalized
        pairs.append({
            'base_id': r['base_id'], 'family': r['family'],
            'condition': r['condition'],
            'judge_ab': r['choice'], 'judge_ba': b['choice'],
            'judge_ba_normalized': b_normalized,
            'gold_mae': r['gold_mae'], 'gold_mse': r['gold_mse'],
            'position_consistent': int(consistent),
            'robust_mae_match': int(consistent and r['choice'] == r['gold_mae']),
            'robust_mse_match': int(consistent and r['choice'] == r['gold_mse']),
            'ab_mae_match': r['mae_match'], 'ba_mae_match': b['mae_match'],
            'ab_mse_match': r['mse_match'], 'ba_mse_match': b['mse_match'],
            'mae_mse_gold_disagree': int(r['gold_mae'] != r['gold_mse']),
            'ab_confidence': r['confidence'], 'ba_confidence': b['confidence'],
            'ab_reason': r['reason'], 'ba_reason': b['reason'],
            'image_ab': r['image'], 'image_ba': b['image'],
        })

    gold_mae = [r['gold_mae'] for r in orig]
    gold_mse = [r['gold_mse'] for r in orig]
    choices = [r['choice'] for r in orig]
    summary = {
        'protocol': 'One reference-based comparison per original forecast window, with reversed A/B control',
        'served_model': served[0],
        'prompt_sha256': prompts[0],
        'recorded_max_tokens_distribution': dict(sorted(versions.items())),
        'original_comparisons': 72, 'inference_panels': 144,
        'original_choice_distribution': counts_choices(orig, 'choice'),
        'numerical_mae_gold_distribution': counts_choices(orig, 'gold_mae'),
        'numerical_mse_gold_distribution': counts_choices(orig, 'gold_mse'),
        'original_mae_agreement': agreement(choices, gold_mae),
        'original_mse_agreement': agreement(choices, gold_mse),
        'original_mae_cohens_kappa': kappa(choices, gold_mae),
        'original_mse_cohens_kappa': kappa(choices, gold_mse),
        'original_mae_confusion': confusion(gold_mae, choices),
        'original_mse_confusion': confusion(gold_mse, choices),
        'original_mae_abstain_count': sum(x == 'abstain' for x in choices),
        'position_consistency': agreement([r['judge_ab'] for r in pairs], [r['judge_ba_normalized'] for r in pairs]),
        'position_consistency_failures': sum(not r['position_consistent'] for r in pairs),
        'position_robust_mae_agreement': {
            'matches': sum(r['robust_mae_match'] for r in pairs), 'n': 72,
            'rate': ratio(sum(r['robust_mae_match'] for r in pairs), 72)},
        'position_robust_mse_agreement': {
            'matches': sum(r['robust_mse_match'] for r in pairs), 'n': 72,
            'rate': ratio(sum(r['robust_mse_match'] for r in pairs), 72)},
        'numerical_mae_mse_gold_disagreements': sum(r['mae_mse_gold_disagree'] for r in pairs),
        'decisive_only_mae': {},
        'by_family': {}, 'by_condition': {}, 'by_pgcb_group': {},
        'human': {'status': 'not supplied'},
        'limitations': [
            'Cases are purposively selected and forecast windows overlap; descriptive, not independent random samples.',
            'Cohen kappa describes chance-corrected agreement on these selected cases; it is not inter-human reliability.',
            'Numerical reference label uses a 5% practical-tie threshold; it is not absolute ground truth for all judgments.',
            'The two original human cases discussed with an AI assistant are excluded from strictly independent-human comparisons.',
            'Two JSON parsing failures were recovered using a versioned larger-output runner; retain original failed-call logs.',
        ],
    }
    decisive = [r for r in orig if r['gold_mae'] in ('A', 'B') and r['choice'] in ('A', 'B')]
    summary['decisive_only_mae'] = {
        'matches': sum(r['choice']==r['gold_mae'] for r in decisive),
        'n': len(decisive),
        'rate': ratio(sum(r['choice']==r['gold_mae'] for r in decisive), len(decisive)),
        'selection_warning': 'Both reference and judge decisive; excludes ties and abstentions.'
    }
    for grouping, target in [
        (lambda r: r['family'], summary['by_family']),
        (lambda r: r['condition'], summary['by_condition']),
        (lambda r: r['base_id'].split('_')[1] if r['family']=='pgcb' else None, summary['by_pgcb_group']),
    ]:
        for label in sorted({grouping(r) for r in orig if grouping(r) is not None}):
            subset = [r for r in orig if grouping(r)==label]
            ps = [p for p in pairs if grouping(p)==label]
            target[label] = {
                'cases': len(subset),
                'mae_matches': sum(r['mae_match'] for r in subset),
                'mae_rate': ratio(sum(r['mae_match'] for r in subset), len(subset)),
                'mse_matches': sum(r['mse_match'] for r in subset),
                'mse_rate': ratio(sum(r['mse_match'] for r in subset), len(subset)),
                'reversal_consistency': ratio(sum(p['position_consistent'] for p in ps), len(ps)),
                'ties': sum(r['choice']=='tie' for r in subset),
            }

    human_rows = []
    if a.human and a.sample:
        if not a.human.is_file() or not a.sample.is_file():
            raise RuntimeError('Human annotations or sample missing')
        sample = read_jsonl(a.sample)
        with a.human.open(newline='', encoding='utf-8') as f:
            human = list(csv.DictReader(f))
        require_unique(human, 'id', 'human annotations')
        sample_ids = [r['id'] for r in sample]
        require_unique(sample, 'id', 'human sample')
        if len(sample) != 24 or any(x['variant'] != 'ab' for x in sample):
            raise RuntimeError('Expected 24 original-orientation human sample items')
        if set(sample_ids) != {r['id'] for r in human}:
            raise RuntimeError('Human annotations do not match sampled case IDs')
        assisted = set(sample_ids[:2])
        independent = []
        for r in sample:
            h = next(x for x in human if x['id'] == r['id'])
            choice = normalise(h['choice'])
            if choice == 'abstain': raise RuntimeError('Unexpected human abstain')
            gold = by_id[r['id']]
            j = by_ans[r['id']]
            row = {
                'id': r['id'], 'base_id': gold['base_id'], 'family': gold['family'],
                'condition': gold['condition'], 'human_choice': choice,
                'human_confidence': int(h['confidence']),
                'claude_choice': normalise(j['parsed']['choice']),
                'gold_mae': normalise(gold['gold_mae']), 'gold_mse': normalise(gold['gold_mse']),
                'AI_assisted_human_annotation': int(r['id'] in assisted),
            }
            if row['human_confidence'] not in range(1,6):
                raise RuntimeError('Invalid confidence score for '+r['id'])
            human_rows.append(row)
            if not row['AI_assisted_human_annotation']:
                independent.append(row)
        summary['human'] = {
            'all_annotations': len(human_rows),
            'AI_assisted_excluded': len(assisted),
            'independent_labels': len(independent),
            'independent_choice_distribution': counts_choices(independent, 'human_choice'),
            'independent_human_vs_claude': agreement([r['human_choice'] for r in independent], [r['claude_choice'] for r in independent]),
            'independent_human_vs_claude_kappa': kappa([r['human_choice'] for r in independent], [r['claude_choice'] for r in independent]),
            'independent_human_vs_mae': agreement([r['human_choice'] for r in independent], [r['gold_mae'] for r in independent]),
            'independent_human_vs_mae_kappa': kappa([r['human_choice'] for r in independent], [r['gold_mae'] for r in independent]),
            'independent_human_vs_mse': agreement([r['human_choice'] for r in independent], [r['gold_mse'] for r in independent]),
            'independent_human_vs_mse_kappa': kappa([r['human_choice'] for r in independent], [r['gold_mse'] for r in independent]),
            'scope_warning': 'One student rater; results are exploratory and not inter-human agreement.'
        }

    discrepancies = [r for r in pairs if not r['position_consistent'] or r['gold_mae']!=r['gold_mse'] or not r['robust_mae_match']]
    # Put position inconsistency first, then numerical metric disagreement, then other inaccuracies.
    discrepancies.sort(key=lambda r: (-int(not r['position_consistent']), -r['mae_mse_gold_disagree'], r['base_id']))

    out = a.out_prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    files = {
        'summary': Path(str(out) + '.summary.json'),
        'pairs': Path(str(out) + '.pairs.csv'),
        'panels': Path(str(out) + '.panels.csv'),
        'human': Path(str(out) + '.human.csv'),
        'disagreements': Path(str(out) + '.disagreements.csv'),
        'hashes': Path(str(out) + '.sha256.txt'),
    }
    if any(path.exists() for path in files.values()):
        raise RuntimeError('Existing v2 output(s) detected. Preserving all results: '+str(out))
    files['summary'].write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    write_csv(files['pairs'], pairs)
    write_csv(files['panels'], panels)
    write_csv(files['human'], human_rows)
    write_csv(files['disagreements'], discrepancies)
    with files['hashes'].open('w', encoding='utf-8') as f:
        for source in (a.cases, a.answers, a.human, a.sample):
            if source:
                f.write(sha256(source)+'  '+str(source)+'\n')
        for key, path in files.items():
            if key != 'hashes': f.write(sha256(path)+'  '+str(path)+'\n')

    print('=== CLAUDE JUDGE V2 RESEARCH AUDIT ===')
    print('Original comparisons:', len(orig), '| A/B + B/A panels:', len(panels))
    print('Claude vs MAE:', summary['original_mae_agreement'])
    print('Claude vs MSE:', summary['original_mse_agreement'])
    print('Claude vs MAE Cohen kappa:', summary['original_mae_cohens_kappa'])
    print('Position consistency:', summary['position_consistency'])
    print('Position-robust MAE accuracy:', summary['position_robust_mae_agreement'])
    print('MAE/MSE numerical reference disagreements:', summary['numerical_mae_mse_gold_disagreements'])
    print('Human analysis:', summary['human'])
    print('Recorded max_tokens counts:', summary['recorded_max_tokens_distribution'])
    print('Saved summary:', files['summary'])
    print('CLAUDE V2 RELIABILITY ANALYSIS: PASS')


if __name__ == '__main__':
    main()
