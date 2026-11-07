#!/usr/bin/env python3
"""COMP8240 judge study v3: offline, read-only, no API calls.

Adds what v1/v2 lack: pair-clustered bootstrap CIs, tie-threshold sensitivity,
accuracy by difficulty (relative MAE gap), confidence calibration, position-bias
direction, and kappa between every rater (judges, human, MAE/MSE references).
Inputs are never modified. Python >=3.9, standard library only.
"""
import argparse, csv, hashlib, json, math, random
from collections import Counter, defaultdict
from pathlib import Path

EPS = 1e-12
CATS = ('A', 'B', 'tie')


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(c):
    c = str(c or '').strip()
    return {'a': 'A', 'b': 'B', 'tie': 'tie', 'abstain': 'abstain'}.get(c.lower())


def gold_at(a, b, t):
    if abs(a - b) <= t * max(a, b, EPS):
        return 'tie'
    return 'A' if a < b else 'B'


def kappa(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a in CATS and b in CATS]
    n = len(pairs)
    if n == 0:
        return None, 0
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in CATS) / n ** 2
    return (None if pe >= 1 else round((po - pe) / (1 - pe), 4)), n


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def rate(flags):
    k, n = sum(flags), len(flags)
    return {'k': k, 'n': n, 'rate': round(k / n, 4) if n else None, 'wilson95': wilson(k, n)}


def cluster_boot(rows, fn, B=5000, seed=8240):
    groups = defaultdict(list)
    for r in rows:
        groups[r['base_id']].append(r)
    keys = sorted(groups)
    if not keys:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in groups[k]]
        v = fn(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return None
    return [round(vals[int(.025 * len(vals))], 4), round(vals[min(len(vals) - 1, int(.975 * len(vals)))], 4)]


def load_human(path, skip_first):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}, []
    keys = {k.lower(): k for k in rows[0]}
    idk = keys.get('id') or keys.get('case_id')
    ck = keys.get('choice') or keys.get('human_choice') or keys.get('label')
    if not idk or not ck:
        raise RuntimeError('Human CSV needs id and choice columns: ' + str(list(rows[0])))
    skipped = [r[idk] for r in rows[:skip_first]]
    out = {}
    for r in rows[skip_first:]:
        c = norm(r[ck])
        if c is None:
            raise RuntimeError('Unknown human label: ' + repr(r[ck]))
        if r[idk] in out:
            raise RuntimeError('Duplicate human ID: ' + r[idk])
        out[r[idk]] = c
    return out, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', required=True, type=Path, help='private cases.jsonl with gold metrics')
    p.add_argument('--judge', action='append', required=True, metavar='NAME=PATH',
                   help='judge JSONL; repeat for several judges')
    p.add_argument('--human', action='append', default=[], metavar='NAME=PATH',
                   help='human CSV (id,choice[,confidence]); repeat for batches/raters')
    p.add_argument('--human-skip-first', type=int, default=0,
                   help='exclude the first N rows of the FIRST human file (e.g. AI-assisted labels)')
    p.add_argument('--out-prefix', required=True, type=Path)
    a = p.parse_args()

    cases = {c['id']: c for c in read_jsonl(a.cases)}
    need = {'id', 'base_id', 'variant', 'family', 'condition', 'A_mae', 'B_mae', 'A_mse', 'B_mse', 'gold_mae', 'gold_mse'}
    for c in cases.values():
        if not need <= set(c):
            raise RuntimeError('cases.jsonl missing fields: ' + str(need - set(c)))
    thr = {round(float(c.get('gold_tie_threshold', .05)), 6) for c in cases.values()}
    if len(thr) != 1:
        raise RuntimeError('Mixed tie thresholds in manifest')
    base_t = thr.pop()

    judges, provenance = {}, {'cases': [str(a.cases), sha(a.cases)]}
    for spec in a.judge:
        name, path = spec.split('=', 1)
        rows = [r for r in read_jsonl(path) if r.get('valid')]
        ids = [r['id'] for r in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(name + ': duplicate valid IDs')
        judges[name] = {r['id']: {'choice': norm(r['parsed']['choice']),
                                  'confidence': r['parsed'].get('confidence'),
                                  'model': r.get('served_model') or r.get('model')} for r in rows}
        provenance[name] = [path, sha(path), len(rows), sorted({str(v['model']) for v in judges[name].values()})]

    humans, skipped_ids = {}, []
    for i, spec in enumerate(a.human):
        name, path = spec.split('=', 1)
        h, sk = load_human(path, a.human_skip_first if i == 0 else 0)
        skipped_ids += sk
        humans.setdefault(name.split(':')[0], {}).update(h)
        provenance['human:' + name] = [path, sha(path), len(h)]
    for name, h in humans.items():
        bad = set(h) - set(cases)
        if bad:
            raise RuntimeError('Human IDs not in manifest: ' + str(sorted(bad)[:5]))

    report = {'protocol': {'tie_threshold_in_manifest': base_t,
                           'human_rows_excluded_as_ai_assisted': skipped_ids,
                           'note': 'Descriptive statistics over selected, possibly overlapping windows. '
                                   'CIs: Wilson (panel level) and 5000x bootstrap clustered by base comparison.'},
              'provenance': provenance, 'judges': {}, 'humans': {}, 'pairwise_kappa_ab_panels': {}}
    tables = []

    for name, J in judges.items():
        rows = []
        for cid, c in cases.items():
            if cid not in J:
                continue
            j = J[cid]
            rel_gap = abs(c['A_mae'] - c['B_mae']) / max(c['A_mae'], c['B_mae'], EPS)
            rows.append(dict(id=cid, base_id=c['base_id'], variant=c['variant'], family=c['family'],
                             condition=c['condition'], choice=j['choice'], conf=j['confidence'],
                             gold_mae=c['gold_mae'], gold_mse=c['gold_mse'], rel_gap=rel_gap,
                             A_mae=c['A_mae'], B_mae=c['B_mae']))
        ab = [r for r in rows if r['variant'] == 'ab']
        res = {'panels': len(rows), 'ab_panels': len(ab),
               'choice_distribution_ab': dict(Counter(r['choice'] for r in ab)),
               'gold_mae_distribution_ab': dict(Counter(r['gold_mae'] for r in ab))}
        for g in ('gold_mae', 'gold_mse'):
            res[g + '_agreement_ab'] = rate([r['choice'] == r[g] for r in ab])
            res[g + '_agreement_ab']['cluster_boot95'] = cluster_boot(
                ab, lambda s, g=g: sum(r['choice'] == r[g] for r in s) / len(s) if s else None)
            res[g + '_agreement_all_panels'] = rate([r['choice'] == r[g] for r in rows])
            res[g + '_agreement_all_panels']['cluster_boot95'] = cluster_boot(
                rows, lambda s, g=g: sum(r['choice'] == r[g] for r in s) / len(s) if s else None)
            k, n = kappa([r['choice'] for r in ab], [r[g] for r in ab])
            res[g + '_kappa_ab'] = {'kappa': k, 'n': n}
        dec = [r for r in ab if r['gold_mae'] in ('A', 'B') and r['choice'] in ('A', 'B')]
        res['decisive_only_mae_agreement_ab'] = rate([r['choice'] == r['gold_mae'] for r in dec])
        # Position analysis on complete pairs.
        by = defaultdict(dict)
        for r in rows:
            by[r['base_id']][r['variant']] = r
        flip = {'A': 'B', 'B': 'A', 'tie': 'tie', 'abstain': 'abstain'}
        full = [v for v in by.values() if {'ab', 'ba'} <= set(v)]
        cons = [flip[v['ab']['choice']] == v['ba']['choice'] for v in full]
        robust = [flip[v['ab']['choice']] == v['ba']['choice'] and v['ab']['choice'] == v['ab']['gold_mae'] for v in full]
        letters = [r['choice'] for r in rows if r['choice'] in ('A', 'B')]
        res['position_consistency'] = rate(cons)
        res['position_robust_mae_accuracy'] = rate(robust)
        res['first_slot_A_share_of_decisive'] = rate([x == 'A' for x in letters])
        # Tie-threshold sensitivity.
        sens = {}
        for t in (0.0, 0.02, 0.05, 0.10, 0.20):
            gl = [gold_at(r['A_mae'], r['B_mae'], t) for r in ab]
            k, n = kappa([r['choice'] for r in ab], gl)
            sens['{:.2f}'.format(t)] = {'agreement': rate([r['choice'] == x for r, x in zip(ab, gl)])['rate'],
                                        'kappa': k, 'gold_ties': gl.count('tie')}
        res['tie_threshold_sensitivity_ab'] = sens
        # Difficulty: decisive-gold panels, binned by relative MAE gap.
        bins = [(0.0, .05), (.05, .15), (.15, .30), (.30, 10.0)]
        diff = {}
        for lo, hi in bins:
            g = [r for r in ab if lo <= r['rel_gap'] < hi]
            diff['{:.0%}-{}'.format(lo, '' if hi > 1 else '{:.0%}'.format(hi))] = {
                'n': len(g), 'mae_agreement': rate([r['choice'] == r['gold_mae'] for r in g])['rate'],
                'judge_tie_rate': rate([r['choice'] == 'tie' for r in g])['rate']}
        res['accuracy_by_relative_mae_gap_ab'] = diff
        cal = {}
        for cf in sorted({r['conf'] for r in ab if r['conf'] is not None}):
            g = [r for r in ab if r['conf'] == cf]
            cal[str(cf)] = {'n': len(g), 'mae_agreement': rate([r['choice'] == r['gold_mae'] for r in g])['rate']}
        res['confidence_calibration_ab'] = cal
        res['by_family_ab'] = {f: rate([r['choice'] == r['gold_mae'] for r in ab if r['family'] == f])
                               for f in sorted({r['family'] for r in ab})}
        res['by_condition_ab'] = {c: rate([r['choice'] == r['gold_mae'] for r in ab if r['condition'] == c])
                                  for c in sorted({r['condition'] for r in ab})}
        report['judges'][name] = res
        for r in rows:
            tables.append(dict(rater=name, **{k: r[k] for k in ('id', 'base_id', 'variant', 'family', 'condition',
                                                                 'choice', 'conf', 'gold_mae', 'gold_mse')},
                               rel_mae_gap=round(r['rel_gap'], 6)))

    for name, H in humans.items():
        ab = [cid for cid in H if cases[cid]['variant'] == 'ab']
        res = {'labels': len(H), 'ab_labels': len(ab), 'distribution': dict(Counter(H.values()))}
        for g in ('gold_mae', 'gold_mse'):
            res[g + '_agreement'] = rate([H[c] == cases[c][g] for c in ab])
            k, n = kappa([H[c] for c in ab], [cases[c][g] for c in ab])
            res[g + '_kappa'] = {'kappa': k, 'n': n}
        report['humans'][name] = res

    raters = {('gold_mae'): {c: v['gold_mae'] for c, v in cases.items() if v['variant'] == 'ab'},
              ('gold_mse'): {c: v['gold_mse'] for c, v in cases.items() if v['variant'] == 'ab'}}
    for n_, J in judges.items():
        raters[n_] = {c: v['choice'] for c, v in J.items() if cases.get(c, {}).get('variant') == 'ab'}
    for n_, H in humans.items():
        raters['human:' + n_] = {c: v for c, v in H.items() if cases[c]['variant'] == 'ab'}
    names = list(raters)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            common = sorted(set(raters[names[i]]) & set(raters[names[j]]))
            k, n = kappa([raters[names[i]][c] for c in common], [raters[names[j]][c] for c in common])
            agree = rate([raters[names[i]][c] == raters[names[j]][c] for c in common])
            report['pairwise_kappa_ab_panels'][names[i] + ' vs ' + names[j]] = {'kappa': k, 'n': n,
                                                                               'agreement': agree['rate']}

    a.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    js = a.out_prefix.with_name(a.out_prefix.name + '.summary.json')
    js.write_text(json.dumps(report, indent=2))
    pc = a.out_prefix.with_name(a.out_prefix.name + '.panels.csv')
    with pc.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(tables[0]))
        w.writeheader(); w.writerows(tables)
    print('=== JUDGE STUDY V3 (offline) ===')
    for name, r in report['judges'].items():
        m = r['gold_mae_agreement_ab']
        print('{}: MAE agreement {}/{} = {} | Wilson {} | cluster-boot {} | kappa {}'.format(
            name, m['k'], m['n'], m['rate'], m['wilson95'], m['cluster_boot95'], r['gold_mae_kappa_ab']['kappa']))
        print('   position consistency {} | order-robust accuracy {} | first-slot share {}'.format(
            r['position_consistency']['rate'], r['position_robust_mae_accuracy']['rate'],
            r['first_slot_A_share_of_decisive']['rate']))
        print('   by MAE gap:', {k: (v['n'], v['mae_agreement']) for k, v in r['accuracy_by_relative_mae_gap_ab'].items()})
        print('   tie-threshold sensitivity:', {k: (v['agreement'], v['kappa']) for k, v in r['tie_threshold_sensitivity_ab'].items()})
        print('   confidence calibration:', {k: (v['n'], v['mae_agreement']) for k, v in r['confidence_calibration_ab'].items()})
    for k, v in report['pairwise_kappa_ab_panels'].items():
        print('kappa', k, v)
    print('Excluded human rows:', skipped_ids)
    print('Saved:', js, '|', pc)
    print('JUDGE STUDY V3: PASS')


if __name__ == '__main__':
    main()
