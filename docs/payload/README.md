# 载荷算子：宿主是模型，不是细胞

宿主是**一个算子/模型实例**，病原是**能改写它输入或参数的载荷**。非负密度、每步一次不动点、
带计量（放大）与门控的边——同一套骨架，五层 + 三段投递链路。

> 细胞级那条线（22 状态 / HIV-1 / 标定）已被方向变更取代，收在 `legacy/bio/`，两边零交叉引用。

## 文档

| 文件 | 内容 |
| --- | --- |
| `DESIGN.md` | 骨架、五层各加什么、为什么宿主是模型、量纲与参数 |
| `EVIDENCE.md` | **每个数字对应哪次运行**，7 套判据 34 条 |
| `LIMITS.md` | 不能说什么：参数没标定、只有一个宿主、拓扑只在阈值附近有效 |
| `README.md` | 这一页：目录、快速开始、文件清单 |

## 五层

```
宿主（1 个模型实例）   payload_host_operator.py   13 状态 / 22 边，9 条判据
  ├─ 多宿主网络        payload_ecology.py         多株竞争 + 平台 + 桥，5 条
  │    └─ 谱系演化      payload_lineage.py         可突变基因组 + 选择，6 条
  │         └─ 重组      payload_recombination.py   位点交换，3 条
  │              └─ 空间  payload_spatial.py         拓扑 → 阈值，3 条
  └─ 投递链路          payload_channels.py         扇出/口径 → 临界成功率，4 条
       ├─ 载荷存活率     payload_survival.py       切块→检索→截断，4 条
```

共用底座：`payload_core.py`（饱和、7 种图、二分阈值、Shannon、判据打印）。

## 快速开始

```bash
python payload_host_operator.py --check       # 9 条：无工具必死、有工具放大、清空上下文后由权重复活
python payload_ecology.py --check             # 5 条：护栏换赢家、补丁不改总量、桥让组成趋同
python payload_lineage.py --check             # 6 条：选择推高复制、护栏拧向隐蔽、多样性坍缩
python payload_recombination.py --check       # 3 条：重组造出新组合，点突变造不出
python payload_spatial.py --threshold --check # 3 条：阈值方向由投递口径决定
python payload_channels.py --check            # 4 条：同扇出下份额比全量难起 5.7 倍
python payload_survival.py --check          # 4 条：块越细越容易切断载荷
```

## 三个最能说明问题的结论

1. **清空上下文与记忆后，载荷从 weights 复活**（峰 0.197）——输入被改写与参数被改写的分界。
2. **护栏不消灭载荷，只换赢家**：全员加固把爆发从 64/64 压到 0/64，但定植仍是 64/64；
   谱系层同样——护栏把 37 个谱系压到 13 个、方向拧向隐蔽株。
3. **阈值方向由投递语义决定，不由拓扑**：同一个无标度图，share 口径阈值 0.0907、broadcast 0.0078。

投递侧链条（**历史值**，测量脚本 `injection_eval.py` 已移除，理由见 `EVIDENCE.md` §9）：
遵从率 25%/10% × 存活率 8%–35% → 有效成功率 0.020–0.088 / 0.008–0.035。
这条链现在只作为记录存在，不再有可运行的判据。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `payload_operator.py` | 算子类（ViralHostOperator + Edge dataclass） |
| `payload_core.py` | 共用底座（无模型逻辑） |
| `payload_host_operator.py` | 宿主算子，多宿主 `[N,13]` |
| `payload_ecology.py` | 多株竞争 + 平台 + 桥 |
| `payload_lineage.py` | 谱系池：突变与选择 |
| `payload_recombination.py` | 位点级重组 |
| `payload_spatial.py` | 拓扑、Allee、临界传播强度 |
| `payload_channels.py` | 投递语义 → 阈值 |
| `payload_survival.py` | 切块/检索/截断存活率 |
| `dist/payload_operator.zip` | 发布包（17 个文件） |
| `legacy/` | 被取代的细胞级算子、三个死文件、旧的分层文档 |

## 已知边界

全部参数按判据调的、只测过一个 1.5B 宿主、拓扑只在阈值附近有结论、重组没有同源限制。
**这些不是免责声明，是 `LIMITS.md` 里逐条写清的失败与缺口。**
