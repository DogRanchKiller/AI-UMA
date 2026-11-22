"""Offline, run from any cwd: python -B analyst_run.py; a new analyst/ is required."""
from pathlib import Path
import hashlib, json, io, runpy, types
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent/'analyst'
S=R/'output/kline_volume_20261122/results'; M=json.loads((S/'manifest.json').read_bytes())
def sha(b): return hashlib.sha256(b).hexdigest()
def check(p,h):
    b=p.read_bytes(); assert sha(b)==h, str(p); return b
def frozen(n,**kw): return pd.read_csv(io.BytesIO(check(S/n,M['output_sha256'][n])),**kw)
def put(n,t): t.to_csv(O/n,index=False,encoding='utf-8-sig',mode='x',lineterminator='\n',date_format='%Y-%m-%d')
def jsonx(n,v):
    with (O/n).open('x',encoding='utf8') as f: json.dump(v,f,indent=2,ensure_ascii=False)
for n,h in M['input_sha256'].items(): check(R/n,h)
runner=R/'scripts/13_kline_volume.py'; check(runner,M['runner_sha256']); K=runpy.run_path(str(runner)); E=K['engine']()
P=frozen('pool.csv',usecols=['date','gate','code','rank','state','near3'],parse_dates=['date']); assert len(P)==12085 and P.date.nunique()==72
Q=frozen('quote_hashes.csv').set_index('code',verify_integrity=True).sha256.to_dict(); stockindex=M['quote_hash_index']; ix=pd.read_csv(io.BytesIO(check(R/stockindex,M['input_sha256'][stockindex]))).set_index('code',verify_integrity=True); assert all(ix.at[c,'sha256']==h for c,h in Q.items())
q=K['Quotes'](Q); ixcode=M['benchmark']; cal=q.calendar(ixcode); assert cal.equals(q.calendar(M['calendar_cross_check'])) and str(cal[-1].date())==M['last_close']
axis=[d for d in sorted(P.date.unique()) if E['observation_dates'](cal,d)[1] is not None]; assert len(axis)==71 and set(P.date)-set(axis)=={pd.Timestamp('2026-11-20')}
O.mkdir(exist_ok=False); pc=['caliper','date','gate','expand_code','expand_rank','contract_code','contract_rank','rank_gap']; plans=[]; den=[]
for gap in (5,10,20):
    orig=E['match_ranks']; match=types.FunctionType(orig.__code__,dict(orig.__globals__,MAX_GAP=gap));
    for d,g in P[P.date.isin(axis)].groupby('date',sort=True):
        assert g['rank'].is_unique and g.code.is_unique and g.gate.nunique()==1
        codes=g.set_index('rank').code; valid=g[g.state.isin(K['ELIGIBLE'])]; a=valid[valid.near3=='expand']['rank']; b=valid[valid.near3=='contract']['rank']; pairs=match(a,b); den.append([gap,d,g.gate.iloc[0],len(a),len(b),len(pairs)])
        for i,j in pairs: plans.append([gap,d,g.gate.iloc[0],codes.at[i],i,codes.at[j],j,abs(i-j)])
plan=pd.DataFrame(plans,columns=pc); dates=pd.DataFrame(den,columns=['caliper','date','gate','eligible_expand','eligible_contract','planned']); old=frozen('plan.csv',parse_dates=['date']); pd.testing.assert_frame_equal(plan.loc[plan.caliper==5,pc[1:]].reset_index(drop=True),old.loc[(old.spec=='near3')&old.date.isin(axis),pc[1:]].reset_index(drop=True)); assert len(dates)==213
put('plan.csv',plan); put('plan_dates.csv',dates); stamps={n:sha((O/n).read_bytes()) for n in ('plan.csv','plan_dates.csv')}; jsonx('plan_lock.json',stamps)
print('Sealed:',stamps,flush=True)
# Outcome phase is reachable only after all three plans have been saved and verified.
locks=json.loads((O/'plan_lock.json').read_bytes()); plan,dates=[pd.read_csv(io.BytesIO(check(O/n,locks[n])),parse_dates=['date']) for n in ('plan.csv','plan_dates.csv')]; results=[]; drows=[]; summary=[]
for gap in (5,10,20):
    gdate=dates[dates.caliper==gap]; local=[]
    for r in gdate.itertuples(index=False):
        a,b=E['observation_dates'](cal,r.date); complete=up=down=0; selected=plan[(plan.caliper==gap)&(plan.date==r.date)]; assert len(selected)==r.planned
        for p in selected.to_dict('records'):
            obs=E['outcome']({'strong_code':p['expand_code'],'control_code':p['contract_code']},a,b,q.open_at,ixcode)
            results.append(dict(p,**{k.replace('strong_','expand_').replace('control_','contract_'):v for k,v in obs.items() if k not in ('strong_code','control_code')})); assert obs['status'] in ('missing','complete')
            if obs['status']=='complete': complete+=1; up+=int(obs['strong_return']>0); down+=int(obs['control_return']>0)
        local.append(dict(r._asdict(),entry=a,exit=b,complete=complete,missing=r.planned-complete,up_expand=up,up_contract=down,expand_rate=up/complete if complete else np.nan,contract_rate=down/complete if complete else np.nan,diff=(up-down)/complete if complete else np.nan))
    df=pd.DataFrame(local); drows.extend(local); lo,hi=E['block_hit_interval'](df['diff']); row=dict(caliper=gap,mature_days=len(df),planned_days=int(df.planned.gt(0).sum()),effective_days=int(df.complete.gt(0).sum()),planned=int(df.planned.sum()),complete=int(df.complete.sum()),missing=int(df.missing.sum()),expand_pct=100*df.expand_rate.mean(),contract_pct=100*df.contract_rate.mean(),diff_pp=100*df['diff'].mean(),ci_low_pp=100*lo,ci_high_pp=100*hi)
    if gap==5: assert (row['effective_days'],row['complete'],round(row['expand_pct'],2),round(row['contract_pct'],2))==(16,87,49.45,51.86); baseline=row.copy()
    row.update({'change_'+k:row[k]-baseline[k] for k in ('planned_days','effective_days','planned','complete','missing','diff_pp')}); summary.append(row)
put('pair_endpoints.csv',pd.DataFrame(results)); put('daily.csv',pd.DataFrame(drows)); put('summary.csv',pd.DataFrame(summary)); put('quotes_used.csv',pd.DataFrame([dict(code=c,sha256=Q[c],purpose='|'.join(sorted(v))) for c,v in sorted(q.used.items())])); jsonx('manifest.json',dict(original_manifest_sha256=sha((S/'manifest.json').read_bytes()),source_pool_sha256=M['output_sha256']['pool.csv'],source_quote_index_sha256=M['output_sha256']['quote_hashes.csv'],source_inputs=M['input_sha256'],research_runner_sha256=M['runner_sha256'],analysis_script_sha256=sha(Path(__file__).read_bytes()),out_sha256={p.name:sha(p.read_bytes()) for p in sorted(O.iterdir())},block=3,draws=10000,seed=20261121,pandas=pd.__version__,numpy=np.__version__)); print(pd.DataFrame(summary).to_string(index=False))
