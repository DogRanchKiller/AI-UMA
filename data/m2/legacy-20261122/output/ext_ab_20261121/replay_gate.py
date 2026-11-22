"""Paired gate replay. Run from workspace root; does not write production outputs."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pandas as pd
s=importlib.util.spec_from_file_location('build','scripts/02_build.py'); b=importlib.util.module_from_spec(s); s.loader.exec_module(b)
r=Path(__file__).resolve().parent; cfg=b.Config.load(Path('output/config_used.json'))
master,names,panel=b.load_inputs(cfg); mats=b.fc.clean_matrices(b.fc.build_matrices(panel)); del panel
close=mats['close']; ds=close.index; bench=close[cfg.benchmark].dropna(); mask=b.uv.build_mask(mats,names,cfg,universe_codes=list(names)); buy,sell=b.uv.tradeable_at_open(mats['open'],close.shift(),list(close.columns)); vols=close.pct_change(fill_method=None).rolling(60,min_periods=40).std(); old=b.bt._execute
summary=[]; rows=[]; curves={}; holdings={}
def known(d,sig,*args,**kw):
    kw['vol_row']=vols.loc[sig]; return old(d,sig,*args,**kw)
for model,path in [('base',Path('output')),('ext',r)]:
    p=pd.read_csv(path/'preds.csv',parse_dates=['date'],dtype={'score':'float32'}); sc=p.pivot(index='date',columns='code',values='score').reindex(columns=close.columns); sc,reg=b.po.build_target_scores(sc,mats,mask,bench,cfg); g=reg.reindex(ds); prev=g.shift(1,fill_value=False); dates=ds[ds.get_indexer(sc.index[:-1])+1]; assert len(dates)==71
    for mode in ['original','known']:
        b.bt._execute=old if mode=='original' else known
        m=mask if mode=='original' else mask.shift(1,fill_value=False); gate=g if mode=='original' else prev
        for gated in [False,True]:
            res=b.bt.run(sc,mats,m,buy,sell,bench,cfg,regime_ok=gate if gated else None); eq=res.equity
            if mode=='original':
                stats=json.loads((path/'run_summary.json').read_text(encoding='utf8'))['performance' if gated else 'performance_raw_signal']; assert abs(res.stats['年化收益']-stats['年化收益'])<0.00001,(res.stats,stats)
            k=(mode,gated,model); curves[k]=eq; holdings[k]=[set(res.holdings_by_date[d]) for d in dates]; summary.append([*k,res.stats['年化收益'],res.stats['最大回撤'],res.stats['夏普比率'],eq.loc['2025'].iloc[-1]/eq.loc[:'2024'].iloc[-1]-1]);
sumdf=pd.DataFrame(summary,columns=['mode','gate','model','annual','drawdown','sharpe','2025']);sumdf.to_csv(r/'gate_replay.csv',index=False);print(sumdf.to_string(index=False));
for mode in ['original','known']:
    gate=g if mode=='original' else prev
    for gated in [False,True]:
        a=curves[mode,gated,'base']; e=curves[mode,gated,'ext']; da=a.diff()/a.loc[:'2024'].iloc[-1]; de=e.diff()/e.loc[:'2024'].iloc[-1]; ar=a.pct_change(); er=e.pct_change()
        for i,d in enumerate(dates):
            ix=ds[(ds>=d)&(ds<(dates[i+1] if i+1<len(dates) else ds[-1]+pd.Timedelta(days=1)))]; x=ix[ix.year==2025]; h=holdings[mode,gated,'base'][i]; j=holdings[mode,gated,'ext'][i]; rows.append([mode,gated,d,bool(gate.loc[d]),len(h&j),len(h),len(j),len(x),(de[x]-da[x]).sum(),np.log1p(er[ix]).sum()-np.log1p(ar[ix]).sum()]);
df=pd.DataFrame(rows,columns=['mode','gate','execution','allowed','common','base_n','ext_n','days_2025','contribution_2025','log_relative']);df.to_csv(r/'gate_periods.csv',index=False); t=df.query("mode=='known' and gate and days_2025>0"); print(t.sort_values('contribution_2025').to_string(index=False));print(df.query("mode=='known' and not gate").groupby('allowed').log_relative.agg(['count','sum']));
