#!/usr/bin/env python3
"""Blinded, resumable, auditable forecast-image judge: OpenAI or Anthropic API.

Input: public_cases.jsonl + images from VLM Judge v3, with no model labels/gold.
Output: compatible with the existing tools/vlm_judge/analyze_judge.py.
Python >=3.9, standard library only. API keys are read from environment only.
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
DEFAULT_MODELS = {'openai': 'gpt-5.4-2026-03-05', 'anthropic': 'claude-sonnet-4-6'}
PRICING_USD_PER_MILLION = {
    'openai': {'input': 2.50, 'output': 15.00},
    'anthropic': {'input': 3.00, 'output': 15.00},
}


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
        raise ValueError('No JSON object returned')
    choice = str(obj.get('choice','')).strip().lower()
    if choice not in ('a','b','tie','abstain'):
        raise ValueError('Invalid judge choice: '+repr(choice))
    confidence = obj.get('confidence')
    if type(confidence) is not int or confidence not in range(1,6):
        raise ValueError('Confidence must be integer 1..5')
    reason = str(obj.get('reason','')).strip()
    if not reason or len(reason)>800:
        raise ValueError('Invalid reason length')
    return {'choice': choice.upper() if choice in ('a','b') else choice,
            'confidence': confidence, 'reason': reason}


def make_request(provider, model, image64):
    if provider == 'openai':
        key = os.getenv('OPENAI_API_KEY')
        if not key: raise RuntimeError('OPENAI_API_KEY is not set')
        url='https://api.openai.com/v1/chat/completions'
        headers={'Authorization': 'Bearer '+key}
        payload={
            'model':model, 'temperature':0, 'reasoning_effort':'none',
            'max_completion_tokens':256, 'response_format':{'type':'json_object'},
            'messages':[
                {'role':'system','content':SYSTEM_PROMPT},
                {'role':'user','content':[
                    {'type':'text','text':USER_PROMPT},
                    {'type':'image_url','image_url':{'url':'data:image/png;base64,'+image64,'detail':'high'}}
                ]},
            ],
        }
    elif provider == 'anthropic':
        key=os.getenv('ANTHROPIC_API_KEY')
        if not key: raise RuntimeError('ANTHROPIC_API_KEY is not set')
        url='https://api.anthropic.com/v1/messages'
        headers={'x-api-key':key,'anthropic-version':'2023-06-01'}
        payload={
            'model':model,'max_tokens':1024,'temperature':0,
            'system':SYSTEM_PROMPT,
            'messages':[{'role':'user','content':[
                {'type':'text','text':USER_PROMPT},
                {'type':'image','source':{'type':'base64','media_type':'image/png','data':image64}},
            ]}],
        }
    else:
        raise RuntimeError('Unsupported provider')
    headers['Content-Type']='application/json'
    return url,headers,payload


def api_call(url,headers,payload,max_attempts=3):
    data=json.dumps(payload, ensure_ascii=False).encode('utf-8')
    for attempt in range(max_attempts):
        request=Request(url, data=data, headers=headers, method='POST')
        try:
            with urlopen(request, timeout=180) as r:
                return json.loads(r.read().decode('utf-8'))
        except HTTPError as exc:
            message=exc.read(2500).decode('utf-8', errors='replace')
            if exc.code not in (408,409,429,500,502,503,504) or attempt == max_attempts-1:
                raise RuntimeError('HTTP {}: {}'.format(exc.code,message)) from None
        except (URLError, TimeoutError) as exc:
            if attempt == max_attempts-1: raise RuntimeError(str(exc)) from None
        time.sleep(min(2**attempt*3,15))
    raise RuntimeError('Unreachable API retries state')


def read_answer(provider, response):
    if provider=='openai':
        answer=response['choices'][0]['message']['content']
        tokens=response.get('usage', {})
        inp=int(tokens.get('prompt_tokens',0)); out=int(tokens.get('completion_tokens',0))
    else:
        answer='\n'.join(item.get('text','') for item in response.get('content', []) if item.get('type')=='text')
        tokens=response.get('usage', {})
        inp=int(tokens.get('input_tokens',0));out=int(tokens.get('output_tokens',0))
    if not isinstance(answer,str): raise ValueError('Non-string API answer')
    return answer,inp,out


def load_records(path):
    if not path.is_file(): raise RuntimeError('File missing: '+str(path))
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(obj,sort_keys=True,ensure_ascii=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def validate_input(cases,root):
    required={'id','base_id','variant','image','image_sha256'}
    if len(cases)!=144: raise RuntimeError('Expected 144 panel entries, found {}'.format(len(cases)))
    if len({x['id'] for x in cases})!=144: raise RuntimeError('Duplicate input case ID')
    for case in cases:
        if set(case)!=required: raise RuntimeError('Blind manifest has extra/missing fields: '+str(case.get('id')))
        p=(root/case['image']).resolve()
        if root.resolve() not in p.parents: raise RuntimeError('Invalid image path traversal')
        if not p.is_file(): raise RuntimeError('Image unavailable: '+str(p))
        if hashlib.sha256(p.read_bytes()).hexdigest()!=case['image_sha256']:
            raise RuntimeError('Image checksum mismatch: '+str(p))
    pairs={}
    for c in cases: pairs.setdefault(c['base_id'],set()).add(c['variant'])
    if len(pairs)!=72 or any(x!={'ab','ba'} for x in pairs.values()):
        raise RuntimeError('Expected 72 complete A/B and B/A pairs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider',choices=sorted(DEFAULT_MODELS),required=True)
    p.add_argument('--model',default=None)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0,help='0 for all remaining; 4 pilot')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--budget-usd',type=float,default=5.0)
    a=p.parse_args()
    model=a.model or DEFAULT_MODELS[a.provider]
    if a.limit<0 or a.budget_usd<=0: p.error('limit >=0 and budget-usd >0 required')
    cases=load_records(a.cases);validate_input(cases,a.cases.parent)
    print('BLIND INPUT VERIFICATION: PASS (144 panels / 72 paired comparisons)',flush=True)
    print('PROVIDER:',a.provider,'MODEL:',model,'PROMPT:',PROMPT_VERSION,flush=True)
    if a.dry_run:
        print('DRY RUN: NO API REQUESTS; NO CHARGES; NO OUTPUT WRITTEN')
        return
    already=load_records(a.output) if a.output.exists() else []
    if len({x['id'] for x in already})!=len(already):
        raise RuntimeError('Existing answer file has duplicated IDs; do not modify')
    if any(x.get('model')!=model or x.get('provider')!=a.provider for x in already):
        raise RuntimeError('Output file contains another provider/model; use a separate file')
    done={x['id'] for x in already if x.get('valid')}
    # Prior successful runs count towards the user-set spend cap.
    spend=sum(float(x.get('estimated_usd',0)) for x in already)
    prompt_sha=hashlib.sha256((PROMPT_VERSION+SYSTEM_PROMPT+USER_PROMPT).encode('utf-8')).hexdigest()
    failures=a.output.with_name(a.output.stem+'_errors.jsonl')
    n=0
    for case in cases:
        if case['id'] in done: continue
        if spend>=a.budget_usd:
            print('SPEND LIMIT REACHED; stopping before next case',flush=True);break
        png=(a.cases.parent/case['image']).read_bytes()
        image64=base64.b64encode(png).decode('ascii')
        url,headers,payload=make_request(a.provider,model,image64)
        started=datetime.datetime.now(datetime.timezone.utc).isoformat()
        response=None; raw=None
        try:
            response=api_call(url,headers,payload)
            raw,inp,out=read_answer(a.provider,response)
            parsed=parse_response(raw)
            rates=PRICING_USD_PER_MILLION[a.provider]
            estimate=(inp*rates['input']+out*rates['output'])/1e6
            record={'id':case['id'],'base_id':case['base_id'], 'variant':case['variant'],
                    'image_sha256':case['image_sha256'],'provider':a.provider,'model':model,
                    'prompt_version':PROMPT_VERSION,'prompt_sha256':prompt_sha,
                    'temperature':0,'max_tokens':1024,'completed_at':started,
                    'raw_response':raw,'parsed':parsed,'valid':True,
                    'response_id':response.get('id'),'served_model':response.get('model'),'usage_input_tokens':inp,
                    'usage_output_tokens':out,'estimated_usd':round(estimate,8)}
            append_jsonl(a.output,record)
            spend+=estimate;done.add(case['id'])
            print('{} VALID {} cumulative_usage_cost≈${:.4f}'.format(case['id'],parsed['choice'],spend),flush=True)
        except Exception as exc:
            append_jsonl(failures, dict(id=case["id"], time=started, error=type(exc).__name__+": "+str(exc)[:1000], stop_reason=(response or {}).get("stop_reason"), raw_response=raw if isinstance(raw,str) else None))
            print('{} ERROR {}'.format(case['id'],str(exc)[:160]),flush=True)
        n+=1
        if a.limit and n>=a.limit: break
    print('NEW REQUESTS:',n,'SUCCESSFUL TOTAL:',len(done),'OF',len(cases),flush=True)
    print('ESTIMATED API CHARGE FROM RECORDED USAGE: ${:.4f}'.format(spend),flush=True)
    print('SAVED:',a.output,flush=True)
    if failures.exists(): print('ERROR LOG:',failures,flush=True)
    if len(done)==144: print('HOSTED JUDGE ALL 144: COMPLETE',flush=True)

if __name__=='__main__': main()
