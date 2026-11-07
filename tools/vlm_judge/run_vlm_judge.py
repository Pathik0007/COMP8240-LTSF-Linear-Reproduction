#!/usr/bin/env python3
"""Execute blinded image comparisons against a localhost OpenAI-compatible VLM."""
import argparse
import base64
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

PROMPT_VERSION = 'comp8240_forecast_visual_judge_v1_2026-11-02'
SYSTEM_PROMPT = '''You are an independent evaluator of numerical time-series forecasts.
Your task is to inspect the supplied chart and compare Forecast A and Forecast B
against the black "Observed future" trace over the future time interval ONLY.
The grey "Observed history" trace is context, not a ground-truth test target.
Judge closeness to future observations including level, trend, timing, spikes,
and reversals. Do not automatically prefer smoother forecasts.
You do not know model identities or error metrics. Never invent them.
Choose exactly one of A, B, tie, abstain. Use tie if the forecasts appear
similarly accurate, abstain if the image is unreadable.
Reply with one JSON object only, exactly this schema:
{"choice":"A", "confidence":3, "reason":"One brief observation grounded in the chart."}
The confidence is an integer 1 to 5. No extra text or markdown.'''
USER_PROMPT = 'Based only on the plot, which forecast more accurately matches the observed FUTURE over the 96-hour horizon? Respond with the required JSON.'


def parse_response(s):
    s = s.strip()
    if s.startswith('```'):
        s = re.sub(r'^```(?:json)?\s*|\s*```$', '', s, flags=re.IGNORECASE).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r'\{[\s\S]*?\}', s)
        obj = json.loads(m.group()) if m else None
    if not isinstance(obj, dict):
        raise ValueError('Model returned no JSON object')
    choice = str(obj.get('choice', '')).strip().lower()
    if choice not in ('a','b','tie','abstain'):
        raise ValueError('Invalid choice: ' + repr(choice))
    confidence = obj.get('confidence')
    if not isinstance(confidence, int) or isinstance(confidence, bool) or confidence not in range(1,6):
        raise ValueError('confidence must be integer 1...5')
    reason = str(obj.get('reason','')).strip()
    if not reason or len(reason) > 800:
        raise ValueError('Missing or overlong reason')
    return {'choice': choice.upper() if choice in ('a','b') else choice,
            'confidence': confidence, 'reason': reason}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--limit', type=int, default=0,
                        help='0=all remaining; 4=quick pilot')
    args = parser.parse_args()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('Run: python -m pip install openai') from exc
    if not args.base_url.startswith(('http://127.0.0.1:', 'http://localhost:')):
        raise RuntimeError('Refusing a non-local server URL. Serve privately on Vast.ai.')
    cases = [json.loads(x) for x in args.cases.read_text().splitlines() if x.strip()]
    existing = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                existing.add(json.loads(line)['id'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompt_sha = hashlib.sha256((PROMPT_VERSION+SYSTEM_PROMPT+USER_PROMPT).encode()).hexdigest()
    client = OpenAI(api_key='LOCAL_INFERENCE_ONLY', base_url=args.base_url, timeout=180, max_retries=1)
    count=0
    for case in cases:
        if case['id'] in existing:
            continue
        path = args.cases.parent / case['image']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != case['image_sha256']:
            raise RuntimeError('Image SHA mismatch: {}'.format(path))
        url = 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')
        result = {'id':case['id'], 'base_id':case['base_id'], 'variant':case['variant'],
                  'image_sha256':case['image_sha256'], 'model':args.model,
                  'prompt_version': PROMPT_VERSION, 'prompt_sha256':prompt_sha,
                  'temperature':0, 'max_tokens':180,
                  'completed_at':datetime.now(timezone.utc).isoformat()}
        try:
            response = client.chat.completions.create(
                model=args.model, temperature=0, max_tokens=180,
                messages=[{'role':'system','content':SYSTEM_PROMPT},
                          {'role':'user','content':[{'type':'text','text':USER_PROMPT},
                            {'type':'image_url','image_url':{'url':url,'detail':'high'}}]}])
            raw = response.choices[0].message.content or ''
            result['raw_response'] = raw
            result['parsed'] = parse_response(raw)
            result['valid'] = True
        except Exception as exc:
            result['valid'] = False
            result['error'] = type(exc).__name__ + ': ' + str(exc)[:500]
        with args.output.open('a') as f:
            f.write(json.dumps(result, sort_keys=True) + '\n')
        print(case['id'], 'VALID' if result['valid'] else 'ERROR',
              result.get('parsed',{}).get('choice',result.get('error')), flush=True)
        count+=1
        if args.limit and count >= args.limit:
            break
    print('Saved {} new judge records to {}'.format(count, args.output))


if __name__ == '__main__':
    main()
