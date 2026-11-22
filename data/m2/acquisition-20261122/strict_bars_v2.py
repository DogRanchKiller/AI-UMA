"""Run once per batch: python -B strict_bars_v2.py. Preserves exact SDK headers/rows before production fetch_kline validates; never manufactures a valid receipt from a failed history."""
import sys
if sys.flags.optimize:
    raise RuntimeError('Raw evidence collection requires Python with optimization disabled; refusing IO.')
import argparse, csv, datetime as dt, gzip, json, os, time
from contextlib import contextmanager
import acquire_v1 as a
from baostock.common import context as bs_context
V=a.P/'verified-acquisition-v2'
BAR=['date','code','open','high','low','close','volume','amount','adjustflag']; FULL_FACTOR=['code','dividOperateDate','foreAdjustFactor','backAdjustFactor','adjustFactor']; USED_FACTOR=['code','dividOperateDate','backAdjustFactor']

class EndSafeSocket:
    def __init__(self,socket): self.socket=socket
    def recv(self,*args):
        data=self.socket.recv(*args)
        if not data:raise ConnectionError('peer closed socket before SDK message terminator')
        return data
    def __getattr__(self,key):return getattr(self.socket,key)

def strict_begin():
    a.begin()
    if not isinstance(bs_context.default_socket,EndSafeSocket):bs_context.default_socket=EndSafeSocket(bs_context.default_socket)

def force_reconnect():
    sock=getattr(bs_context,'default_socket',None)
    if sock is not None:sock.close()
    a.LIVE=False

def consume(response,expected):
    head=list(response.fields)
    if response.error_code!='0' or head!=expected:raise ValueError(('source status/schema',response.error_code,response.error_msg,head,expected))
    rows=[]
    while response.next():
        row=response.get_row_data()
        if len(row)!=len(head) or not all(isinstance(v,str) for v in row):raise ValueError('SDK row width or string type differs')
        rows.append(list(row))
    if response.error_code!='0' or (response.data and len(response.data)==int(response.per_page_count)):raise ValueError(('incomplete pagination',response.error_code,response.error_msg))
    return dict(fields=head,rows=rows)

def network(code):
    strict_begin();dotted=code[:2]+'.'+code[2:]
    q=[consume(a.bs.query_history_k_data_plus(dotted,','.join(BAR),start_date=a.START,end_date=a.END,frequency='d',adjustflag='3'),BAR)]
    if q[0]['rows'] and code not in a.ds.REFERENCE_INDICES:q.append(consume(a.bs.query_adjust_factor(dotted,start_date='1990-01-01',end_date=a.END),FULL_FACTOR))
    return dict(code=code,source=a.SRC,start=a.START,end=a.END,adjust_start='1990-01-01',queries=q)

def replay(code,source):
    # Delegate every numeric transformation and its range validation to project fetch_kline; only SDK transport is replayed from the independently retained exact rows.
    d=code[:2]+'.'+code[2:];qs=source['queries'];expected=[BAR]+([FULL_FACTOR] if qs and qs[0]['rows'] and code not in a.ds.REFERENCE_INDICES else [])
    if (source['source'],source['code'],source['start'],source['end'],source['adjust_start'])!=(a.SRC,code,a.START,a.END,'1990-01-01') or [q['fields'] for q in qs]!=expected:raise ValueError('cache identity/window/schema mismatch')
    for q,fe in zip(qs,expected):
        if not isinstance(q['rows'],list) or any(len(row)!=len(fe) or not all(isinstance(v,str) for v in row) or row[fe.index('code')]!=d for row in q['rows']):raise ValueError('cache row shape/type/symbol mismatch')
    class Transport:
        def query_history_k_data_plus(self,*args,**kwargs):return qs[0]
        def query_adjust_factor(self,*args,**kwargs):return qs[1]
    @contextmanager
    def session():yield Transport()
    def project(q,wanted):
        fields=q['fields']
        if list(wanted)!=(BAR if fields==BAR else USED_FACTOR):raise ValueError('unexpected project consumer fields')
        return [tuple(row[fields.index(f)] for f in wanted) for row in q['rows']]
    prev=a.ds._bs_rows,a.ds._bs_session;a.ds._bs_rows,a.ds._bs_session=project,session
    try:return a.ds.fetch_kline(code,a.Config(),start=dt.date.fromisoformat(a.START),end=dt.date.fromisoformat(a.END))
    finally:a.ds._bs_rows,a.ds._bs_session=prev

def one(code):
    sdk=V/'sdk-raw'/(code+'.json.gz'); fp=V/'raw'/'baostock_raw_v1'/(code+'.csv');rp=V/'receipts'/(code+'.json'); did_network=not sdk.exists()
    if did_network:
        source=network(code);raw=gzip.compress(a.J(source),compresslevel=1,mtime=0);a.store(sdk,raw)
    else:raw=sdk.read_bytes();source=json.loads(gzip.decompress(raw))
    fr=replay(code,source)
    if fr.empty:raise ValueError('SDK returned empty bars')
    dates=fr.date.dt.strftime('%Y-%m-%d').tolist();data=fr.to_csv(index=False,lineterminator='\n',date_format='%Y-%m-%d').encode();r=dict(code=code,source=a.SRC,start=a.START,end=a.END,rows=len(fr),first=dates[0],last=dates[-1],dates_sha256=a.H(''.join(x+'\n' for x in dates).encode()),csv_sha256=a.H(data),sdk_sha256=a.H(raw),csv_path=fp.relative_to(V).as_posix(),sdk_path=sdk.relative_to(V).as_posix());a.store(fp,data);a.store(rp,a.J(r));return 'queried' if did_network else 'reverified'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--codes',nargs='+');args=parser.parse_args()
    history=[r['code'] for r in csv.DictReader((V/'legacy-history.csv').open())];assert len(history)==len(set(history))==5226
    wanted=args.codes or history
    if any(c not in history for c in wanted):raise ValueError('not in frozen scope')
    stamp=dt.datetime.now().strftime('%Y%m%dT%H%M%S%f');num_fail=0
    with (V/f'bars-{stamp}.ndjson').open('x',encoding='utf8',buffering=1) as log:
        for code in wanted:
            for attempt in (1,2):
                try:status=one(code);error=None
                except Exception as exc:
                    status='failed';error=repr(exc)
                    if not (V/'sdk-raw'/(code+'.json.gz')).exists():force_reconnect()
                log.write(json.dumps(dict(at=dt.datetime.now().astimezone().isoformat(),code=code,attempt=attempt,status=status,error=error),ensure_ascii=False)+'\n')
                if not error:break
                if (V/'sdk-raw'/(code+'.json.gz')).exists() or attempt==2:num_fail+=1;break
                time.sleep(.5)
    force_reconnect();return bool(num_fail)
if __name__=='__main__':raise SystemExit(main())
