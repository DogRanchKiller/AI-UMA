"""只读冻结输入，在 data/m2/runs/<run-id>/ 中离线重算；默认仅检查。"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import importlib.metadata
import importlib.util
import io
import json
import math
import re
import socket
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M2 = ROOT / 'data' / 'm2'
ARCHIVE = M2 / 'legacy-20261122' / 'output'
SOURCE = M2 / 'acquisition-20261122' / 'verified-acquisition-v2'
BASE_HASH = {
    'run_summary.json': 'fb96419b4126e09516f262a9d4257f8cf1e7218f9d77bfd729451dedf0a26c6d',
    'config_used.json': 'd8e7dc43a1a8d1092ed52bad33bac4acf5bba7942bfedd57295f7c17ad0238be',
    'backtest_equity.csv': '14e76f28018bb6a41b253b4cdb8a419eea49a151c14a70c28addf0249a87e6e0',
    'preds.csv': '6e559ad94ee75fe6e671e74a2b4924edf30ae211a2309263f259422914892072',
    'importance.csv': '6c4c03cceb125a4991d972c90746f226bdcb9b675055e369111a71f961ceb0c6',
}
META_HASH = {
    'master_list.csv': '2c3f05e2cb332e002ea21ccd6080beb8638751a9fbca1fa5521d345880511f7c',
    'industry.csv': '6d4e45df0eb28208c792744325988c48a64f22b7146270316563589b85791757',
    'legacy-history.csv': 'd4bddee570aadebd5f96db310538f99254662e6ba7f0081e595c4c600ef5652f',
    'legacy-dates.csv.gz': 'cea6ec925181744598f7ac50047378f79799901dad27a1865e75350d5f8ec008',
    'CONTRACT-v1.md': 'a34afc4bde55750cd254254132569252162eb528dba584c406cd0a982f84ae9f',
    'calendar.csv': '1ac0e1b1bc412a219940bb1c8b89a580666e2aebcc16cd8d775ff4e142b3ff9a',
    'target-roster.json': '9b947131ad54077e165d2c6bb4c37cbe40a36f5dacfb2db9671d8b040e662799',
    'BARS-PROVENANCE-v2.md': '0e223650146c543263d05d62c47e75941e159c22fe18822c884f2de90a018e83',
    'frozen-meta.csv': 'fcf9a0c98afc8bd96f422a60a1efd3a44fcf216276c2ce371abf720431fa4777',
}
SOURCE_ID = 'BaoStock SDK 0.9.4/public-api.baostock.com:10030'
START, END = '2019-01-01', '2026-11-20'
SDK_BARS = 'date,code,open,high,low,close,volume,amount,adjustflag'.split(',')
SDK_FACTORS = ['code', 'dividOperateDate', 'foreAdjustFactor', 'backAdjustFactor', 'adjustFactor']
DIV = ('code,dividPreNoticeDate,dividAgmPumDate,dividPlanAnnounceDate,dividPlanDate,'
       'dividRegistDate,dividOperateDate,dividPayDate,dividStockMarketDate,dividCashPsBeforeTax,'
       'dividCashPsAfterTax,dividStocksPs,dividCashStock,dividReserveToStockPs').split(',')
READY_FILES = {'bars_manifest_sha256': 'bars-manifest.csv', 'dividend_coverage_sha256': 'dividend-coverage.csv',
               'dividend_events_sha256': 'dividend-events.csv', 'halt_evidence_sha256': 'halt-evidence.csv'}
INDICES = ('sh000300', 'sh000905', 'sh000852', 'sh000001')


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def read_blob(path, expected=None):
    if path.resolve() != path:
        raise ValueError(f'拒绝重定向的输入路径: {path}')
    blob = path.read_bytes()
    if expected is not None and digest(blob) != expected:
        raise ValueError(f'冻结文件 SHA256 不匹配: {path}')
    return blob


def pinned(directory, hashes):
    return {name: read_blob(directory / name, expected) for name, expected in hashes.items()}


def check_values(record, expected, label):
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f'{label}: {key} 与冻结契约不一致')


def manifest_table(blob, cols, keys):
    df = table(blob, dtype=str, keep_default_na=False)
    if list(df) != cols or df.duplicated(keys).any():
        raise ValueError(f'清单列不符或 {keys} 重复')
    return df


def release(directory):
    marker = read_blob(directory / 'READY.json')
    ready = json.loads(marker)
    check_values(ready, {'version': 'm2-acquisition-v2', 'status': 'complete',
                         'contract_sha256': META_HASH['CONTRACT-v1.md'],
                         'bars_provenance_sha256': META_HASH['BARS-PROVENANCE-v2.md'],
                         'frozen_meta_sha256': META_HASH['frozen-meta.csv']}, 'READY.json')
    hashes = {f: ready[k] for k, f in READY_FILES.items()}
    if not all(isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h) for h in hashes.values()):
        raise ValueError('READY 缺少有效文件摘要')
    return pinned(directory, hashes), {**ready, 'ready_sha256': digest(marker)}


def date_hash(dates):
    return digest(''.join(d + '\n' for d in dates).encode('ascii'))


def sdk_rows(query, columns, label):
    rows = query['rows']
    if (query['fields'] != columns or not isinstance(rows, list) or
            any(not isinstance(r, list) or len(r) != len(columns) or
                not all(isinstance(v, str) for v in r) for r in rows)):
        raise ValueError(f'{label}: SDK 字段或原始行不符合规范')
    return rows


def read_actions(directory, coverage_blob, events_blob, stocks, years):
    """完整矩形＋原始14字符串核对：同一事件可完整重复，不允许挑选冲突版本。"""
    coverage = manifest_table(coverage_blob, ['code', 'year', 'status', 'rows', 'raw_path', 'raw_sha256', 'detail'], ['code', 'year'])
    expected = set(itertools.product(stocks, map(str, years)))
    if set(zip(coverage.code, coverage.year)) != expected:
        raise ValueError('独立权益查询缺年份/股票，或超出冻结覆盖范围')
    unique = {}
    for rec in coverage.itertuples():
        name = f'dividend-queries/{rec.code}/{rec.year}.json'
        if rec.status not in ('empty', 'queried') or rec.raw_path != name:
            raise ValueError(f'权益查询未完成或路径不符: {rec.code}/{rec.year}')
        q = json.loads(read_blob(directory / name, rec.raw_sha256))
        dotted = f'{rec.code[:2]}.{rec.code[2:]}'
        check_values(q, {'source': SOURCE_ID, 'code': dotted, 'year': rec.year, 'yearType': 'operate',
                         'status': rec.status}, name)
        rows = sdk_rows(q, DIV, name)
        if str(len(rows)) != rec.rows or (not rows) != (rec.status == 'empty'):
            raise ValueError(f'{name}: 查询状态/行数与源证据不符')
        for r in rows:
            ex = r[DIV.index('dividOperateDate')]
            if r[0] != dotted or not re.fullmatch(rec.year + r'-\d\d-\d\d', ex) or pd.Timestamp(ex).strftime('%Y-%m-%d') != ex:
                raise ValueError(f'{name}: 证券代码或实施年度不符')
            key = (rec.code, ex)
            if unique.setdefault(key, r) != r:
                raise ValueError(f'同键权益记录冲突: {key}')
    events = table(events_blob, dtype=str, keep_default_na=False)
    if list(events) != DIV or events.values.tolist() != [unique[k] for k in sorted(unique)]:
        raise ValueError('权益发布表与逐股逐年度原始查询不一致')
    return events, set(unique)



def table(blob, **opts):
    return pd.read_csv(io.BytesIO(blob), **opts)


def effective_config(old, config_type, end):
    values = {**old, 'end_date': end, 'use_ext_factors': False, 'start_year': 2020}
    if set(values) != {f.name for f in fields(config_type)}:
        raise ValueError('配置键已改变，拒绝隐式继承新默认值')
    return config_type(**values)


def run_directory(run_id):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', run_id):
        raise ValueError('run-id 仅允许小写 ASCII 字母、数字、连字符、下划线')
    dest = M2 / 'runs' / run_id
    if dest.resolve() != dest or not M2.is_dir():
        raise ValueError('隔离目录不能重定向或位于缺失的 M2 目录')
    dest.parent.mkdir(exist_ok=True)
    dest.mkdir(exist_ok=False)
    return dest


def sdk_match(frame, packet, code, event_keys, required):
    """由原 SDK 行重算单位、乘数生效区间，不能仅信 receipt 自称 W2。"""
    check_values(packet, {'source': SOURCE_ID, 'code': code, 'start': START, 'end': END,
                          'adjust_start': '1990-01-01'}, code)
    queries = packet['queries']
    index = code in INDICES
    if len(queries) != (1 if index else 2):
        raise ValueError(f'{code} SDK 原始查询不全')
    raw = pd.DataFrame(sdk_rows(queries[0], SDK_BARS, code), columns=SDK_BARS)
    dotted = f'{code[:2]}.{code[2:]}'
    dates = frame.index.strftime('%Y-%m-%d').tolist()
    if not raw.code.eq(dotted).all() or not raw.adjustflag.eq('3').all() or raw.date.tolist() != dates:
        raise ValueError(f'{code} SDK 日期、代码或不复权标识不符')
    nums = ['open', 'high', 'low', 'close', 'volume', 'amount']
    want = raw[nums].astype(float)
    want.volume /= 100
    if not np.allclose(want.values, frame[nums].values, rtol=0, atol=0):
        raise ValueError(f'{code} 不复权行情/成交量单位/实际金额与 SDK 不一致')
    expected = np.ones(len(frame))
    if not index:
        original = [(r[0], r[1], r[3]) for r in sdk_rows(queries[1], SDK_FACTORS, code)]
        if any(c != dotted or not '1990-01-01' <= day <= END for c, day, v in original):
            raise ValueError(f'{code} 因子来源日期/代码不符')
        series = pd.Series({day: float(v) for c, day, v in original}, dtype=float).sort_index()
        if (len(series) != len(original) or not np.isfinite(series).all() or series.le(0).any()):
            raise ValueError(f'{code} 因子日期重复或数值无效')
        if not series.empty:
            expected = series.reindex(dates, method='ffill').fillna(1.0).to_numpy()
            for day, value in series.items():
                if pd.Timestamp(day).strftime('%Y-%m-%d') != day:
                    raise ValueError(f'{code} 因子生效日期无效')
            changed = series.ne(series.shift(fill_value=1.0))
            for day in series.index[changed]:
                if required and START <= day <= END and (code, day) not in event_keys:
                    raise ValueError(f'{code} {day} 复权乘数变化未匹配独立实施条款')
    if not np.array_equal(expected, frame.adj_factor.to_numpy()):
        raise ValueError(f'{code} 当日已生效累计因子不符')


def halt_checker(directory, blob, calendar):
    files = manifest_table(blob, ['day', 'raw_path', 'raw_sha256'], ['day'])
    evidence = {}
    for r in files.itertuples():
        name = f'halt-evidence/{r.day}.json'
        if r.day not in calendar or r.raw_path != name:
            raise ValueError('停牌证据日期或路径不符')
        q = json.loads(read_blob(directory / name, r.raw_sha256))
        check_values(q, {'source': SOURCE_ID, 'day': r.day}, name)
        rows = sdk_rows(q, ['code', 'tradeStatus', 'code_name'], name)
        if not rows or len({a for a, b, c in rows}) != len(rows):
            raise ValueError(f'{name}: 名册缺失或代码重复')
        evidence[r.day] = ({a for a, b, c in rows if b == '0'}, r.raw_sha256)

    def proof(code, day):
        halted, sha = evidence.get(day, (set(), None))
        return sha if f'{code[:2]}.{code[2:]}' in halted else None
    return proof


def project_dates(frame, spec, old_dates, proof, exclusions):
    dates = frame.index.strftime('%Y-%m-%d').tolist()
    if len(old_dates) != spec.rows or date_hash(old_dates) != spec.dates_sha256:
        raise ValueError(f'{spec.code} 冻结旧日期证据不一致')
    old = set(old_dates)
    if old.difference(dates):
        raise ValueError(f'{spec.code} 真实日线缺旧日 {min(old.difference(dates))}')
    for day in sorted(set(dates) - old):
        bar = frame.loc[pd.Timestamp(day)]
        sha = proof(spec.code, day)
        if spec.code in INDICES or bar.volume != 0 or bar.amount != 0 or not sha:
            raise ValueError(f'{spec.code} {day} 额外日期无权威停牌证据或非零量额/指数')
        exclusions.append({'code': spec.code, 'date': day, 'volume': 0, 'amount': 0,
                           'tradeStatus': '0', 'halt_raw_sha256': sha})
    projected = frame.loc[pd.to_datetime(old_dates)]
    if date_hash(projected.index.strftime('%Y-%m-%d')) != spec.dates_sha256:
        raise ValueError(f'{spec.code} 比较视图日期与冻结日历不一致')
    return projected.reset_index(names='date')


def panel_view(frame, spec, dates, proof, exclusions):
    # 同样检查未入旧面板的 34 只，不能让一只新增真实交易日跨过 120 门槛后仍被排除。
    view = project_dates(frame, spec, dates, proof, exclusions)
    eligible = len(view) >= 120  # scripts/02_build.py load_inputs(..., min_rows=120)
    if spec.panel != int(eligible):
        raise ValueError(f'{spec.code} 有效交易行数与冻结入池标志不匹配')
    return view if eligible else None


def load_bar(directory, manifest, spec, ds, event_keys):
    code = spec.code
    name, sdk = f'raw/baostock_raw_v1/{code}.csv', f'sdk-raw/{code}.json.gz'
    receipt = json.loads(read_blob(directory / f'receipts/{code}.json', manifest.receipt_sha256))
    check_values(receipt, {'code': code, 'source': SOURCE_ID, 'start': START, 'end': END,
                           'csv_path': name, 'sdk_path': sdk, 'csv_sha256': manifest.csv_sha256,
                           'sdk_sha256': manifest.sdk_sha256}, f'{code} receipt')
    bars = table(read_blob(directory / name, manifest.csv_sha256), float_precision='round_trip')
    if list(bars) != ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'adj_factor', 'code']:
        raise ValueError(f'{code} 行情字段不匹配')
    frame = ds._dated_bars(bars, code)
    ds._validate_bars(frame, code in INDICES)
    dates = frame.index.strftime('%Y-%m-%d').tolist()
    if not dates or dates[0] < START or dates[-1] > END:
        raise ValueError(f'{code} 空行情或超出冻结窗口')
    facts = {'rows': len(frame), 'dates_sha256': date_hash(dates), 'first': dates[0], 'last': dates[-1]}
    check_values(receipt, facts, code)
    check_values(manifest._asdict(), {**{k: str(v) for k, v in facts.items()},
                 'legacy_dates_match': str(int(facts['dates_sha256'] == spec.dates_sha256))}, code)
    packet = json.loads(gzip.decompress(read_blob(directory / sdk, manifest.sdk_sha256)))
    sdk_match(frame, packet, code, event_keys, required=bool(spec.panel))
    return frame


def load_inputs(meta, files, directory, ds, old_dates):
    master = table(meta['master_list.csv'])
    names = dict(zip(master.code.map(ds.to_tencent), master.name))
    industry = table(meta['industry.csv'])
    industry = industry[~industry.industry.isin(['', 'None', 'nan'])]
    history = table(meta['legacy-history.csv'])
    specs = {row.code: row for row in history.itertuples()}
    codes = list(dict.fromkeys([*names, *INDICES]))
    stocks = {c for c in specs if specs[c].panel and c not in INDICES}
    if set(codes) != set(specs) or len(codes) != 5226 or len(stocks) != 5188:
        raise ValueError('冻结股票身份与元数据不一致')
    events, keys = read_actions(directory, files['dividend-coverage.csv'], files['dividend-events.csv'], stocks, range(2018, 2027))
    manifests = manifest_table(files['bars-manifest.csv'], ['code', 'receipt_sha256', 'csv_sha256', 'sdk_sha256',
                               'dates_sha256', 'rows', 'first', 'last', 'legacy_dates_match'], ['code'])
    by_code = {r.code: r for r in manifests.itertuples()}
    if set(by_code) != set(specs):
        raise ValueError('行情清单证券与冻结股票身份不同')
    calendar = table(meta['calendar.csv']).date.tolist()
    if calendar != old_dates:
        raise ValueError('市场日历与冻结主基线净值日历不相等')
    proof = halt_checker(directory, files['halt-evidence.csv'], set(calendar))
    # 按代码流式解压 ~1000 万个冻结日期，而非在内存保留数百万个 Python 字符串。
    panel, exclusions, seen = {}, [], set()
    with gzip.open(io.BytesIO(meta['legacy-dates.csv.gz']), 'rt', encoding='ascii', newline='') as f:
        reader = csv.reader(f)
        if next(reader) != ['code', 'date']:
            raise ValueError('冻结旧日期字段不匹配')
        for code, rows in itertools.groupby(reader, key=lambda row: row[0]):
            if code not in specs or code in seen:
                raise ValueError(f'未知或重复的日期分组 {code}')
            seen.add(code)
            spec = specs[code]
            dates = [day for _, day in rows]
            frame = load_bar(directory, by_code[code], spec, ds, keys)
            view = panel_view(frame, spec, dates, proof, exclusions)
            if view is not None:
                panel[code] = view
    if seen != set(specs) or set(panel) != stocks | set(INDICES):
        raise ValueError('冻结面板或非面板来源覆盖不完整')
    # build_matrices 保留输入列顺序；还原旧 master+indices 顺序，不能改为字典序。
    panel = {code: panel[code] for code in codes if code in panel}
    return (master, names, panel), dict(zip(industry.code, industry.industry)), events, exclusions


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def verify_output(old, blobs, cfg, new, out):
    for name in ('run_summary.json', 'signal.json', 'factors_meta.json', 'preds.csv', 'importance.csv',
                 'backtest_equity.csv', 'weights.csv', 'ic.csv', 'layers.csv'):
        if not (out / name).is_file() or not (out / name).stat().st_size:
            raise ValueError(f'构建缺少产物: {name}')
    if new['diagnostics']['成交笔数'] and not (out / 'backtest_trades.csv').is_file():
        raise ValueError('成交明细文件缺失')
    if json.loads((out / 'run_summary.json').read_bytes()) != new:
        raise ValueError('正式落盘摘要与运行返回不一致')
    if new['config'] != cfg.to_dict() or new['universe']['date_range'] != old['universe']['date_range']:
        raise ValueError('新旧回测区间或有效配置不匹配')
    for group, key in [('universe', 'stocks_in_panel'), ('model', 'n_factors'), ('performance', '交易日数')]:
        if old[group][key] != new[group][key]:
            raise ValueError(f'新旧口径不匹配: {group}.{key}')
    for file, col, ordered in [('backtest_equity.csv', 'date', True), ('importance.csv', 'factor', False), ('preds.csv', 'date', False)]:
        previous = table(blobs[file], usecols=[col])[col].tolist()
        current = pd.read_csv(out / file, usecols=[col])[col].tolist()
        if not (previous == current if ordered else set(previous) == set(current)):
            raise ValueError(f'新旧 {file} 的 {col} 不匹配')
    comparison = {}
    for group in ('performance', 'performance_raw_signal', 'performance_baseline_factor', 'ic', 'ic_baseline_factor', 'diagnostics'):
        comparison[group] = {}
        for key, before in old[group].items():
            after = new[group][key]
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (before, after)):
                raise ValueError(f'{group}.{key} 非有限数值')
            comparison[group][key] = {'old': before, 'new': after, 'delta': after - before}
    return comparison


def execute(dest, build):
    blobs = pinned(ARCHIVE, BASE_HASH)
    old, values = (json.loads(blobs[f]) for f in ('run_summary.json', 'config_used.json'))
    if old['config'] != values:
        raise ValueError('归档配置与主基线不匹配')
    meta = pinned(SOURCE, META_HASH)
    files, ready = release(SOURCE)
    # config 导入时 mkdir；没有原有目录便中止，绝不在旧路径创建目录。
    if not all((ROOT / p).is_dir() for p in ('data/raw', 'data/cache', 'output')):
        raise ValueError('原工作区目录不完整，拒绝导入有 mkdir 副作用的配置模块')
    sys.path.insert(0, str(ROOT))
    from ashare import datasource as ds
    from ashare.config import Config

    start, end = old['universe']['date_range']
    cfg = effective_config(values, Config, end)
    inputs, industry, actions, exclusions = load_inputs(meta, files, SOURCE, ds, table(blobs['backtest_equity.csv']).date.tolist())
    write_json(dest / 'omitted_halts.json', exclusions)
    manifest = {'baseline': BASE_HASH, 'metadata': META_HASH, 'ready': ready,
                'panel_stocks': len(inputs[2]) - len(INDICES), 'indices': list(INDICES),
                'query_pairs': 5188 * 9, 'unique_actions': len(actions), 'cash_basis': 'gross/pre-tax',
                'omitted_halts': len(exclusions), 'omitted_halts_sha256': digest((dest / 'omitted_halts.json').read_bytes()),
                'config': cfg.to_dict(), 'market_window': [start, end], 'python': sys.version,
                'packages': {k: importlib.metadata.version(k) for k in ('numpy', 'pandas', 'lightgbm')},
                'code_sha256': {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in
                                [*sorted((ROOT / 'ashare').glob('*.py')), ROOT / 'scripts/02_build.py', Path(__file__).resolve()]}}
    write_json(dest / 'inputs.json', manifest)
    if not build:
        return 'preflight_passed'
    spec = importlib.util.spec_from_file_location('m2_build', ROOT / 'scripts/02_build.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_build(module, dest, old, blobs, cfg, inputs, industry, actions)
    return 'built'


def run_build(module, dest, old, blobs, cfg, inputs, industry, actions):
    stage, cache = dest / 'unvalidated', dest / 'cache'
    stage.mkdir()
    cache.mkdir()
    # 私有 02_build 实例：不用 ds.load_panel/master_list/industry 下载回退；即使该入口以后
    # 多出一次意外的网络访问也拒绝连接。factors.pkl、preds.pkl 的写入仅指向新 cache。
    module.OUT_DIR, module.CACHE_DIR = stage, cache
    module.load_inputs = lambda config: inputs
    module.ds = SimpleNamespace(industry_map=lambda config: industry)
    with (patch.object(socket.socket, 'connect', side_effect=RuntimeError('M2 禁止联网')),
          patch.object(socket.socket, 'connect_ex', side_effect=RuntimeError('M2 禁止联网'))):
        _, summary = module.build_pipeline(cfg, use_cache=False, corporate_actions=actions)
    comparison = verify_output(old, blobs, cfg, summary, stage)
    write_json(stage / 'comparison.json', comparison)
    write_json(stage / 'config_used.json', cfg.to_dict())
    stage.rename(dest / 'output')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True, help='不可复用的目录名')
    ap.add_argument('--build', action='store_true', help='预检通过后启动全量重算')
    args = ap.parse_args(argv)
    try:
        dest = run_directory(args.run_id)
    except (OSError, ValueError) as exc:
        print(f'M2 未开始: {exc}', file=sys.stderr)
        return 2
    try:
        status = execute(dest, args.build)
    except (Exception, SystemExit) as exc:
        write_json(dest / 'status.json', {'status': 'blocked', 'error': str(exc)})
        print(f'M2 中止，不发布对比: {exc}', file=sys.stderr)
        return 2
    write_json(dest / 'status.json', {'status': status})
    print(f'{status}: {dest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
