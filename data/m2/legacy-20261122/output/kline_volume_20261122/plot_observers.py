"""绘制两只既有观察股的冻结日K，不用于选样。需要 Pillow 和微软雅黑；--out 应指向新文件。"""
import argparse
import csv
import hashlib
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ASOF = '2026-11-20'
CODES = {'sh600795': '国电电力', 'sh601963': '重庆银行'}
HASHES = {
    'sh000300': 'd6b3ef74b6a7fac295174aa21a1ded6d32823cb7d8e3114156271a227e3d2c98',
    'sh600795': '023574df064247d0b2d2576ccc468a80425dbc07a0e3df86ab6593ffab20cf2d',
    'sh601963': '6d2bccb89d4ab9e8079ed65cc423f4c24158977df03f48845aa8b54ae90f37c3',
}
INK, GRAY, GRID = '#172e46', '#506274', '#dce3e8'
UP, DOWN, HI, BAND, AVG = '#b94732', '#147564', '#35698b', '#e8f2f8', '#815628'


def rows(code):
    path = ROOT / 'data' / 'raw' / (code + '.csv')
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[code]:
        raise ValueError(f'冻结行情不符：{code}')
    with path.open(encoding='utf-8-sig', newline='') as source:
        return {r['date']: r for r in csv.DictReader(source) if r['date'] <= ASOF}


def plot(path):
    calendar = sorted(rows('sh000300'))[-119:]
    assert len(calendar) == 119 and calendar[-1] == ASOF
    image = Image.new('RGB', (1440, 840), 'white')
    d = ImageDraw.Draw(image)
    fonts = Path(os.environ['WINDIR']) / 'Fonts'

    def text(x, y, s, size=19, ink=INK, bold=False, anchor='la'):
        font = ImageFont.truetype(str(fonts / ('msyhbd.ttc' if bold else 'msyh.ttc')), size)
        d.text((x, y), s, fill=ink, font=font, anchor=anchor)

    text(48, 30, '同样放量，收盘位置不同', 41, bold=True)
    text(50, 95, '两只原观察股  /  截至 2026-11-20 收盘  /  最近 60 个交易日', 21, GRAY)
    d.line((48, 146, 1392, 146), fill=GRID, width=2)
    d.line((719, 166, 719, 723), fill=GRID)
    for col, (code, name) in enumerate(CODES.items()):
        x0 = 48 + 692 * col
        bars = rows(code)
        f = {k: [float(bars[day][k]) for day in calendar] for k in ('open', 'high', 'low', 'close', 'volume')}
        assert all(0 < lo <= min(o, c) <= max(o, c) <= h and v > 0 for lo, o, c, h, v in
                   zip(f['low'], f['open'], f['close'], f['high'], f['volume']))
        highs = [max(f['high'][i-59:i+1]) for i in range(59, 119)]
        floors = [.97*h for h in highs]
        avgs = [sum(f['volume'][i-20:i])/20 for i in range(59, 119)]
        cl, h, v = f['close'][-1], highs[-1], f['volume'][-1]
        text(x0, 162, name, 29, bold=True)
        text(x0+150, 173, code[2:], 18, GRAY)
        text(x0, 211, f'收盘 {cl:.2f} 元  /  60日高点 {h:.2f} 元  /  距高点 {(1-cl/h)*100:.2f}%', 20)
        inside = cl >= floors[-1]
        text(x0, 249, ('在高点 3% 范围内' if inside else '不在高点 3% 范围内') + f'　·　当日量为此前20日均量的 {v/avgs[-1]:.2f} 倍', 18, HI, True)
        left, right, top, bottom = x0+56, x0+637, 306, 533
        def xx(j):
            return left + (j+.5)*(right-left)/60
        mn, mx = min(min(f['low'][-60:]), min(floors)), max(highs)
        margin = .075*(mx-mn) or 1.; lo, hi = mn-margin, mx+margin
        def yy(p):
            return bottom - (p-lo)/(hi-lo)*(bottom-top)
        d.polygon([(xx(j), yy(p)) for j,p in enumerate(highs)] +
                  [(xx(j), yy(floors[j])) for j in range(59,-1,-1)], fill=BAND)
        for i in range(4):
            p = lo+(hi-lo)*i/3; y = yy(p)
            d.line((left, y, right, y), fill=GRID)
            text(left-8,y,f'{p:.2f}',16,GRAY,anchor='rm')
        d.line([(xx(j), yy(p)) for j,p in enumerate(highs)], fill=HI, width=2)
        d.line([(xx(j), yy(p)) for j,p in enumerate(floors)], fill='#87adc3', width=2)
        vt, vb = 609, 691
        vmax = 1.05*max(max(f['volume'][-60:]),max(avgs))
        def vy(amount):
            return vb-(vb-vt)*amount/vmax
        text(x0, 560, '成交量（万手）', 16, GRAY)
        text(right, 560, f'当日 {v/10000:.1f}  /  此前20日均量 {avgs[-1]/10000:.1f}', 16, GRAY, anchor='ra')
        d.line((left,vt,right,vt), fill=GRID); d.line((left,vb,right,vb), fill=GRID)
        text(left-8,vt,f'{vmax/10000:.0f}',15,GRAY,anchor='rm'); text(left-8,vb,'0',15,GRAY,anchor='rm')
        for j in range(60):
            i = j+59; x=xx(j); o,c=f['open'][i],f['close'][i]; colr=UP if c >= o else DOWN
            d.line((x, yy(f['high'][i]), x, yy(f['low'][i])), fill=colr, width=1)
            a,b=sorted((yy(o),yy(c))); d.rectangle((x-2.6,a,x+2.6,max(a+1,b)),fill=colr)
            d.rectangle((x-2.7,vy(f['volume'][i]),x+2.7,vb),fill=colr)
        d.line([(xx(j),vy(a)) for j,a in enumerate(avgs)],fill=AVG,width=2)
        for j in (0, 29, 59):
            text(xx(j),702,calendar[j+59][5:],16,GRAY,anchor='ma')
    d.line((48, 748, 1392, 748), fill=GRID, width=2)
    text(50, 766, '浅蓝带：各日当时的60日最高价至其下方3%　　量图棕线：各日此前20日均量（不含当日）　　红 / 绿：收盘不低于 / 低于当日开盘', 17, GRAY)
    text(50, 803, '两只股票来自既有观察卡，不是研究中的一对对照；图示为前复权日线，不代表未来表现。', 17, GRAY)
    with path.open('xb') as dest:
        image.save(dest, format='PNG', optimize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('两只观察股的量价位置.png'))
    plot(parser.parse_args().out)
