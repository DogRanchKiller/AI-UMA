"""Run from project root: python -B data/m2/acquisition-20261122/acquire_v1.py bars|actions [workers=2]. Immutable receipts, read-only public queries, per-process serial SDK. Retry by rerunning after fixing reported errors. No release marker is produced here."""
import argparse, csv, datetime as dt, gzip, hashlib, json, os, socket, sys, time, uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[2])); import baostock as bs
from ashare import datasource as ds
from ashare.config import Config
from contextlib import contextmanager
START='2019-01-01'; END='2026-11-20'; SRC='BaoStock SDK 0.9.4/public-api.baostock.com:10030'; DIV=['code','dividPreNoticeDate','dividAgmPumDate','dividPlanAnnounceDate','dividPlanDate','dividRegistDate','dividOperateDate','dividPayDate','dividStockMarketDate','dividCashPsBeforeTax','dividCashPsAfterTax','dividStocksPs','dividCashStock','dividReserveToStockPs']; RAW=P/'raw/baostock_raw_v1'; REC=P/'receipts'; EV=P/'dividend-queries'; PAY=P/'sdk-raw'; LIVE=False
H=lambda b:hashlib.sha256(b).hexdigest()
J=lambda d:json.dumps(d,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')
def store(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=b:raise RuntimeError('immutable content mismatch: '+str(p))
  return
 tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.part')
 with tmp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 tmp.rename(p) # Windows rename is exclusive

def begin():
 global LIVE
 if not LIVE:
  socket.setdefaulttimeout(15);r=bs.login()
  if r.error_code!='0':raise RuntimeError(f'login {r.error_code}:{r.error_msg}')
  LIVE=True

def end():
 global LIVE
 if LIVE:
  try:bs.logout()
  finally:LIVE=False

def fetch(code):
 rp=REC/(code+'.json');fp=RAW/(code+'.csv');sp=PAY/(code+'.json.gz')
 if rp.exists():
  r=json.loads(rp.read_bytes());assert H(fp.read_bytes())==r['csv_sha256'] and H(sp.read_bytes())==r['sdk_sha256'];return 'cached'
 begin();cap=[];orig=ds._bs_rows;session=ds._bs_session
 @contextmanager
 def active():yield bs
 def rows(rs,fields):
  result=orig(rs,fields);cap.append(dict(fields=list(fields),rows=result));return result
 ds._bs_session=active;ds._bs_rows=rows
 try: fr=ds.fetch_kline(code,Config(),start=dt.date.fromisoformat(START),end=dt.date.fromisoformat(END))
 finally:ds._bs_session=session;ds._bs_rows=orig
 if fr.empty:raise ValueError('SDK returned empty bars')
 dates=fr.date.dt.strftime('%Y-%m-%d').tolist();source=gzip.compress(J(dict(code=code,source=SRC,start=START,end=END,adjust_start='1990-01-01',queries=cap)),compresslevel=1,mtime=0);data=fr.to_csv(index=False,lineterminator='\n',date_format='%Y-%m-%d').encode();store(sp,source);store(fp,data)
 r=dict(code=code,source=SRC,start=START,end=END,rows=len(fr),first=dates[0],last=dates[-1],dates_sha256=H(''.join(d+'\n' for d in dates).encode()),csv_sha256=H(data),sdk_sha256=H(source),csv_path=fp.relative_to(P).as_posix(),sdk_path=sp.relative_to(P).as_posix());store(rp,J(r));return 'queried'

def actions(code):
 if code in ds.REFERENCE_INDICES:return 'index'
 sp=PAY/(code+'.json.gz');sr=json.loads((REC/(code+'.json')).read_bytes());b=sp.read_bytes();assert H(b)==sr['sdk_sha256'];q=json.loads(gzip.decompress(b))['queries'];assert len(q)==2
 # Include preceding calendar year to identify any payment/listing spilling into the frozen window, and current end year for boundary coverage.
 years=sorted({'2018','2026'}|{r[1][:4] for r in q[1]['rows'] if START<=r[1]<=END});bs_code=code[:2]+'.'+code[2:]
 for y in years:
  path=EV/code/(y+'.json')
  if path.exists():
   prior=json.loads(path.read_bytes());assert prior['code']==bs_code and prior['year']==y and prior['fields']==DIV;continue
  begin();r=bs.query_dividend_data(bs_code,year=y,yearType='operate');raw=ds._bs_rows(r,DIV);assert all(row[0]==bs_code for row in raw)
  store(path,J(dict(source=SRC,code=bs_code,year=y,yearType='operate',fields=DIV,rows=raw,status='queried' if raw else 'empty')))
 return 'queried'

def do(item):
 mode,code=item
 try:return code,(fetch(code) if mode=='bars' else actions(code)),None
 except Exception as e:
  try:end()
  except Exception:pass
  return code,'failed',repr(e)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['bars','actions']);ap.add_argument('workers',type=int,nargs='?',default=2);a=ap.parse_args();assert 1<=a.workers<=4
 wanted=[r['code'] for r in csv.DictReader((P/'legacy-history.csv').open())];assert len(wanted)==5226 and len(set(wanted))==5226
 stamp=dt.datetime.now().strftime('%Y%m%dT%H%M%S%f');log=P/(a.mode+'-'+stamp+'.ndjson');items=[(a.mode,c) for c in wanted];counts={}
 with log.open('x',encoding='utf8',buffering=1) as f, ProcessPoolExecutor(max_workers=a.workers) as ex:
  for n,(c,s,e) in enumerate(ex.map(do,items,chunksize=1),1):
   counts[s]=counts.get(s,0)+1;f.write(json.dumps(dict(at=dt.datetime.now().astimezone().isoformat(),code=c,status=s,error=e),ensure_ascii=False)+'\n');
   if n%200==0: print(n,counts,flush=True)
 print(counts,flush=True);return int(bool(counts.get('failed')))
if __name__=='__main__':raise SystemExit(main())
