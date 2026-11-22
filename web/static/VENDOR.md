# 第三方前端依赖来源说明（vendored）

本目录下的第三方前端库以**文件形式入库**，运行时不访问任何 CDN。
理由：看板要能在断网的本机直接打开；CDN 不可用时会整页白屏，而这套界面是研究流程的
唯一出口，不能依赖外网可用性。

| 文件 | 版本 | 体积（字节） | SHA-256 |
|---|---|---|---|
| `echarts.min.js` | Apache ECharts 5.6.0 | 1,030,855 | `E84270BD0CD5BDF60FEFC26D00C2A391CB2E81F4D26A7A9EE16185A54773A3CF` |
| `echarts-gl.min.js` | echarts-gl 2.1.0 | 639,846 | `D6D60F2165CE3EB1EA360EF62E8490835E0AE2C1E0C1788A772FFF918E350842` |
| `echarts-gl.LICENSE.txt` | echarts-gl 2.1.0 附带的 LICENSE 原文 | 1,510 | `55EA01207028F76D844678511F29FC800F8A1E67A8A0FE80470128677847AD32` |

## 来源

- 上游仓库：<https://github.com/ecomfe/echarts-gl>，发布版本 `2.1.0`，
  npm 记录的 `gitHead` 为 `2bd401e7ca5e10cdfd96643c9b646fd7ee6ca04b`。
- 取回地址：`https://cdn.jsdelivr.net/npm/echarts-gl@2.1.0/dist/echarts-gl.min.js`
  与 `https://cdn.jsdelivr.net/npm/echarts-gl@2.1.0/LICENSE`。
- 取回时间：2026-11-22（本地时间），取回后已与上游逐字节比对，SHA-256 一致。
- `echarts-gl@2.1.0` 的 peerDependency 为 `echarts ^5.1.2 || ^6.0.0`，本地宿主为 5.6.0，满足要求。

## 许可证（注意：与 package.json 声明不一致）

- `echarts-gl@2.1.0` 的 `package.json` 里写的是 `"license": "MIT"`，
  但该版本**随包附带的 LICENSE 原文是 BSD 3-Clause License，Copyright (c) 2017, Baidu Inc.**
- 本项目按随包 LICENSE 处理：`echarts-gl.LICENSE.txt` 是该文件的**逐字节副本**，
  不得修改；压缩包本身没有 license banner，所以必须靠这份文件满足署名要求。
- `echarts.min.js`（Apache ECharts）为 Apache License 2.0，版权归 Apache Software Foundation。

## 校验方法

```bash
# 逐文件核对本地字节与上表是否一致
python -c "import hashlib,pathlib;[print(p.name, hashlib.sha256(p.read_bytes()).hexdigest().upper()) for p in map(pathlib.Path, ['web/static/echarts.min.js','web/static/echarts-gl.min.js','web/static/echarts-gl.LICENSE.txt'])]"
```

`.gitattributes` 里对这两个 vendor 文件设置了 `-text`，禁止换行转换，保证检出后哈希不变。

## 升级步骤

1. 改 `echarts-gl.min.js` 前先确认宿主 ECharts 版本仍落在其 peerDependency 区间内。
2. 从同一路径取新版本文件与 LICENSE，替换后重新计算 SHA-256 并更新本表。
3. 逐字节比对：`sha256sum` 与 jsdelivr 上的文件一致才算完成，不要用"页面能显示"作为升级依据。
4. 重新跑一遍 3D 视觉验证（见 `DESIGN.md` §12 与 `REVIEW.md` §10）。

## 已知限制

- `echarts-gl@2.1.0` 的 `surface`（surface3D）在本机 headless + SwiftShader 环境下
  只画坐标网格、不画曲面；`bar3D / line3D / scatter3D` 正常。因此本项目不用曲面系列。
- 该库不暴露 `echarts.gl` 全局对象，能力探测只能靠"真建一次离屏 `grid3D`"来判断，
  见 `web/static/index.html` 的 `probeGL()`。
