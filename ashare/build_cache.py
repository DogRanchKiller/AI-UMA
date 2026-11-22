"""构建缓存按实际输入复用；只从本地受信任的缓存目录读取 pickle。"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)
_TRAILER = b'\nASHARE_INPUT_SHA256_V1:'


def _digest(value) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _frame_digest(frame: pd.DataFrame) -> str:
    """对已载入的数值宽表（含日期、代码及次序）分块取指纹，不另读可能已变化的行情文件。"""
    dates = pd.DatetimeIndex(frame.index)
    values = frame.to_numpy(copy=False)
    if values.dtype.kind not in 'bifu':
        raise TypeError('构建缓存只支持数值或布尔宽表')
    desc = (frame.shape, str(dates.dtype), list(frame.columns), values.dtype.str)
    h = hashlib.sha256(bytes.fromhex(_digest(desc)))
    h.update(np.ascontiguousarray(dates.asi8))
    for start in range(0, len(values), 128):
        h.update(np.ascontiguousarray(values[start:start + 128]))
    return h.hexdigest()


def factor_key(mats: dict, cfg: Config, industry: dict, extra: dict) -> str:
    return _digest({
        'schema': 'factors-5-adjusted', 'benchmark': cfg.benchmark,
        'prices': {k: _frame_digest(mats[k]) for k in ('open', 'close', 'high', 'low', 'volume', 'amount', 'adj_factor')},
        'industry': industry if cfg.use_industry_factors else None,
        'external': {k: _frame_digest(v) for k, v in extra.items()} if cfg.use_ext_factors else None,
    })


def prediction_key(factors: str, mask: pd.DataFrame, days, cfg: Config) -> str:
    # 日期、实际股票池以及训练参数决定打分；持仓数、仓位闸门等下游参数不决定打分。
    return _digest({
        'schema': 'stock-index-open-1', 'factors': factors, 'eligible': _frame_digest(mask),
        'dates': [d.isoformat() for d in days], 'horizon': cfg.label_horizon,
        'train_days': cfg.train_days, 'retrain_every': cfg.retrain_every,
        'rounds': cfg.num_boost_round, 'params': cfg.lgb_params,
    })


def read_cache(path: Path, key: str):
    """先读文件尾的小键再决定是否反序列化；旧版缓存无尾标，正常视作未命中。"""
    tail = _TRAILER + key.encode('ascii')
    try:
        with path.open('rb') as f:
            if f.seek(0, 2) < len(tail):
                return None
            f.seek(-len(tail), 2)
            if f.read() != tail:
                return None
            f.seek(0)
            return pickle.load(f)
    except FileNotFoundError:
        return None
    except (OSError, EOFError, ValueError, pickle.UnpicklingError, ImportError, AttributeError, IndexError, OverflowError) as exc:
        log.warning('无法读取构建缓存 %s，重新计算：%s', path.name, type(exc).__name__)
        return None


def write_cache(path: Path, key: str, value) -> None:
    # 这两个信息总是一起替换，中断时不会让旧键指向新内容。仍兼容直接 pickle.load 的读者。
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False) as f:
        tmp = Path(f.name)
    try:
        with tmp.open('wb') as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.write(_TRAILER + key.encode('ascii'))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
