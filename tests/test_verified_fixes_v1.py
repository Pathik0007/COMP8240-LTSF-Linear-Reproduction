#!/usr/bin/env python3
"""Offline fixtures for the separate post-hoc PGCB screening + comparison code."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


class PatchFixtures(unittest.TestCase):
    def test_pgcb_screen_and_temporal_windows(self):
        x=module(ROOT/'src/pgcb_valid_window_benchmark_v2.py','pgcbv2_test')
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'pgcb.csv';n=12000
            d=pd.DataFrame(dict(datetime=pd.date_range('2021-01-01',periods=n,freq='h'),
                                was_observed_hour=True,demand_mw=9000.,generation_mw=8500.,
                                demand_mw_scale_anomaly=False,generation_mw_scale_anomaly=False))
            d.loc[10100,'demand_mw']=74100.
            d.loc[10000,'demand_mw']=73.
            d.loc[9000,'generation_mw']=64526500.
            d.loc[8000,'demand_mw_scale_anomaly']=True
            d.loc[7000,'was_observed_hour']=False
            d.to_csv(p,index=False)
            _,_,v=x.load_data(p)
            self.assertEqual(int(v.sum()),n-5)
            starts,_,_=x.find_starts(v,n,336,96)
            self.assertTrue(all(len(s)>0 for s in starts.values()))
            prefix=np.r_[0,np.cumsum(v.astype(int))]
            for starts_for_split in starts.values():
                self.assertTrue(np.all(prefix[starts_for_split+432]-prefix[starts_for_split]==432))

    def test_common_window_scales_and_cohorts(self):
        with tempfile.TemporaryDirectory() as temp:
            project=Path(temp)
            for tag,starts in (('v1',[10,12,14,16]),('v2',[12,14,16])):
                r=project/'experiments/new_data'/('pgcb_valid_windows_'+tag)
                r.mkdir(parents=True)
                scale=np.array([1000.,2000.]) if tag=='v1' else np.array([1100.,2400.])
                mu=np.array([8000.,7500.]) if tag=='v1' else np.array([8100.,7600.])
                (r/'protocol.json').write_text(json.dumps(dict(features=['demand_mw','generation_mw'],
                    seq_len=336,pred_len=96,train_only_scaler_scale=scale.tolist(),train_only_scaler_mean=mu.tolist())))
                truth_raw=np.array([[[float(s*3+8000),7000.] for _ in range(96)] for s in starts])
                truth=(truth_raw-mu)/scale
                np.savez_compressed(r/'test_reference.npz',starts=np.array(starts),truth_scaled=truth,
                    seasonal24_scaled=truth+.2,seasonal168_scaled=truth+.3)
                for m in ('Linear','NLinear','DLinear'):
                    np.savez_compressed(r/(m+'_test_predictions.npz'),prediction_scaled=truth+.1,
                                        truth_scaled=truth)
            cmd=[sys.executable,str(ROOT/'src/compare_pgcb_v1_v2.py'),'--project',str(project)]
            r=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,msg=r.stdout+r.stderr)
            self.assertIn('PGCB COMMON-WINDOW COMPARISON: PASS',r.stdout)
            manifest=json.loads((project/'results/new_data/pgcb_v1_v2_common_window_v1.json').read_text())
            self.assertEqual(manifest['common_windows'],3)
            self.assertLess(manifest['max_native_truth_difference_mw'],.1)


if __name__=='__main__': unittest.main()
