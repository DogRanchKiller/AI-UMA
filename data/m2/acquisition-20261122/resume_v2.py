"""Bounded, restartable front end for the two immutable v2 collectors. No release marker. Windows/Python 3.12. Never run two owners for one stage."""
import sys
if sys.flags.optimize or not sys.dont_write_bytecode:
    raise RuntimeError('Use unoptimized python -B; refusing IO.')
import argparse, csv, datetime as dt, hashlib, json, os, subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
V=P/'verified-acquisition-v2'
HASHES={P/'acquire_v1.py':'8f7fce2a089055b596bea10e22170796a9fd141a51dacd30b043c6434bf022a7',P/'strict_bars_v2.py':'722b988047a19f15c60ac99b8a6b026013eacad89839dd7b4612ca8eed0e81f2',P/'full_grid_v1.py':'104abf19a83b4269120b0799a67a6d2477d615bccf36058769c4a0fd43a50c7b',V/'BARS-PROVENANCE-v2.md':'0e223650146c543263d05d62c47e75941e159c22fe18822c884f2de90a018e83',V/'frozen-meta.csv':'fcf9a0c98afc8bd96f422a60a1efd3a44fcf216276c2ce371abf720431fa4777'}

def check():
    for f,h in HASHES.items():
        if f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise ValueError(f'collector/metadata mismatch {f}')
    with (V/'frozen-meta.csv').open(encoding='utf8',newline='') as fp:rows=list(csv.DictReader(fp))
    if len(rows)!=7:raise ValueError('not seven authenticated metadata files')
    for row in rows:
        f=V/row['path']
        if f.parent!=V or f.is_symlink() or f.stat().st_size!=int(row['bytes']) or hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:raise ValueError(f'metadata mismatch: {f}')
    print('verified: 3 pinned collectors, base contract plus six independent metadata files, v2 provenance and metadata manifest',flush=True)

def stamp():return dt.datetime.now().astimezone().isoformat(timespec='milliseconds')

class SliceEnd(BaseException):pass

def worker(mode,limit):
    import strict_bars_v2 as bars
    import full_grid_v1 as grid
    import acquire_v1 as a
    n=0
    def before_request():
        nonlocal n
        if limit and n>=limit:raise SliceEnd()
        n+=1
    if mode=='bars':
        original=bars.network
        def guarded(code):
            before_request();return original(code)
        bars.network=guarded
    else:
        # The verified historical grid implementation reads a.P for the roster and logs.
        # Bind that location to v2; no old roster, dividends, or prices can enter via the wrapper.
        a.P=V
        old_begin=a.begin
        def guarded():
            before_request();old_begin();context=bars.bs_context
            if not isinstance(context.default_socket,bars.EndSafeSocket):context.default_socket=bars.EndSafeSocket(context.default_socket)
        a.begin=guarded
    try:
        sys.argv=sys.argv[:1];status=int((bars.main if mode=='bars' else grid.main)())
        print(f'{stamp()} stage={mode} full scan finished rc={status} network work units={n}; never READY',flush=True);return status
    except SliceEnd:
        print(f'{stamp()} stage={mode} bounded new-work limit={limit} reached; remaining inputs PENDING, never READY',flush=True);return 10
    finally:bars.force_reconnect()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['check','bars','equity']);p.add_argument('--max-network',type=int,default=0,help='0 = full scope; positive = bound fresh attempts, not coverage');p.add_argument('--seconds',type=int,default=21600,help='hard child wall-clock bound');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=p.parse_args()
    if args.max_network<0 or args.seconds<1:raise ValueError('invalid work/time bound')
    check()
    if args.mode=='check':return 0
    if args.worker:return worker(args.mode,args.max_network)
    import msvcrt
    runtime=V/'runtime';runtime.mkdir(exist_ok=True)
    with (runtime/(args.mode+'.guard')).open('a+b') as guard:
        if not guard.tell():guard.write(b'1');guard.flush()
        guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
        try:
            path=runtime/(dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'-'+args.mode+'.ndjson')
            with path.open('x',encoding='utf8',buffering=1) as ledger:
                cmd=[sys.executable,'-B',str(Path(__file__).resolve()),args.mode,'--max-network',str(args.max_network),'--worker']
                proc=subprocess.Popen(cmd);start=stamp()
                def event(state,**extra):
                    record=dict(at=stamp(),stage=args.mode,owner='data/task-5',pid=proc.pid,parent_pid=os.getpid(),state=state,started_at=start,timeout=args.seconds,max_network=args.max_network,argv=cmd,**extra);ledger.write(json.dumps(record,ensure_ascii=False)+'\n');print(json.dumps(record,ensure_ascii=True),flush=True)
                event('started');timed_out=False
                try:rc=proc.wait(timeout=args.seconds)
                except subprocess.TimeoutExpired:
                    timed_out=True;proc.kill();proc.wait();rc=124
                except BaseException:
                    proc.kill();proc.wait();event('interrupted',child_returncode=proc.returncode);raise
                event('time_budget_reached' if timed_out else 'slice_budget_reached' if rc==10 else 'full_scan_finished_with_failures' if rc==1 else 'full_scan_returned' if rc==0 else 'worker_failed',exit_code=rc,child_returncode=proc.returncode)
                return rc
        finally:guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_UNLCK,1)
if __name__=='__main__':raise SystemExit(main())
