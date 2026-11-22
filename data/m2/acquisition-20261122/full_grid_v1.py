"""Independent full operate-year grid after priority batch, not a release gate. Run with python -B."""
import sys
if sys.flags.optimize:
    raise RuntimeError('Raw evidence collection requires Python with optimization disabled; refusing IO.')
import csv, datetime as dt, json, time
import acquire_v1 as a
# A separate immutable version is required: v1 priority receipts lost original schema provenance.
ROOT=a.P/'verified-acquisition-v2'

def main():
    panel = [r['code'] for r in csv.DictReader((a.P/'legacy-history.csv').open()) if r['panel']=='1' and r['code'] not in a.ds.REFERENCE_INDICES]
    assert len(set(panel)) == len(panel) == 5188
    stamp=dt.datetime.now().strftime('%Y%m%dT%H%M%S%f'); fails=0
    with (a.P/f'grid-{stamp}.ndjson').open('x',encoding='utf8',buffering=1) as log:
        for code in panel:
            dotted=code[:2]+'.'+code[2:]
            for year in map(str,range(2018,2027)):
                path=ROOT/'dividend-queries'/code/(year+'.json')
                for attempt in (1,2):
                    try:
                        if path.exists():
                            record=json.loads(path.read_bytes()); status='cached'
                        else:
                            a.begin(); response=a.bs.query_dividend_data(dotted,year=year,yearType='operate'); assert response.error_code=='0',(response.error_code,response.error_msg);assert list(response.fields)==a.DIV
                            rows=[]
                            while response.next():
                                raw=response.get_row_data(); assert len(raw)==len(a.DIV) and all(isinstance(v,str) for v in raw), 'source row width/type'; rows.append(list(raw))
                            last=getattr(response,'data',None); assert response.error_code=='0' and not(last and len(last)==int(response.per_page_count)),'incomplete final page'
                            record=dict(source=a.SRC,code=dotted,year=year,yearType='operate',fields=list(response.fields),rows=rows,status='queried' if rows else 'empty');status=record['status']
                        assert record['source']==a.SRC and record['code']==dotted and record['year']==year and record['yearType']=='operate' and record['fields']==a.DIV and record['status']==('queried' if record['rows'] else 'empty')
                        assert all(len(r)==14 and all(isinstance(v,str) for v in r) and r[0]==dotted and dt.date.fromisoformat(r[6]).isoformat()==r[6] and str(dt.date.fromisoformat(r[6]).year)==year for r in record['rows']), 'wrong row width/type/code or operate year'
                        if status!='cached':a.store(path,a.J(record))
                        error=None
                    except Exception as exc:
                        status='failed';error=repr(exc);a.end();time.sleep(.5)
                    log.write(json.dumps(dict(at=dt.datetime.now().astimezone().isoformat(),code=code,year=year,attempt=attempt,status=status,error=error),ensure_ascii=False)+'\n')
                    if status!='failed':break
                else:fails+=1
    a.end();return bool(fails)
if __name__=='__main__':raise SystemExit(main())
