# TrainingGraphFactory

> 来源：factory-everything/factory-everything-项目总表.md ｜ 板块：factory-everything · L3 ｜ 落地约束：4060 8G 单机可跑

## 一、条目信息（原始字段，完整保留）

| 字段 | 内容 |
| --- | --- |
| 项目 | TrainingGraphFactory |
| 类型 | Graph |
| 真正有意思的核心 | 训练全过程 DAG / 状态 / 资源可视化 |

---

## 二、核心命题

**这个项目不是 Training UI。**

它应该成为：**Training Runtime 的数字孪生 / 可观察计算图。**

## 三、三种图必须区分

| # | 图 | 回答的问题 |
| --- | --- | --- |
| 1 | Job DAG | 训练任务由哪些步骤构成，依赖关系如何？ |
| 2 | Training State Graph | 训练当前处于什么状态，异常后如何恢复？ |
| 3 | Resource Graph | 每个步骤占用了哪些资源，瓶颈在哪？ |

三者混在一起就会退化成一张杂乱的流程图，所以必须在数据层就分开。

## 四、整体架构

```
                    Training Run
                         │
                         ▼
                    Event Stream
                         │
         ┌───────────────┼────────────────┐
         ▼               ▼                ▼
      Job Event       Metric Event    Resource Event
         │               │                │
         ▼               ▼                ▼
      DAG Builder      State Store     Resource Graph
         │               │                │
         └───────────────┼────────────────┘
                         ▼
                    TrainingGraphFactory
```

## 五、Job DAG

```
Load Model
    ↓
Load Data
    ↓
Preprocess
    ↓
Train
    ├── checkpoint
    ├── evaluate
    └── log
          ↓
        Export
```

## 六、State Graph

```
INIT
 ↓
PREPARING
 ↓
TRAINING
 ├─→ EVALUATING
 ├─→ CHECKPOINTING
 ├─→ PAUSED
 └─→ FAILED
          ↓
       RECOVER
```

## 七、Resource Graph

```
Job
 ├─ GPU0
 ├─ GPU1
 ├─ CPU
 ├─ RAM
 ├─ VRAM
 ├─ Disk
 └─ Network
```

## 八、真正有意思的是把异常链路关联起来

```
Loss spike
       ↓
GPU utilization
       ↓
Data batch
       ↓
Checkpoint
       ↓
Config
```

**最终任何异常都可以回答：**

> What happened? When? On which resource? Under which config? To which checkpoint?

这是它区别于 TensorBoard 类工具的关键：不是画曲线，而是**在五张不同的信号之间建立可追溯的因果链**。

## 九、阶段性落地计划

- **M0｜事件流骨架**：定义 Job / Metric / Resource 三类 Event 的 schema，跑通采集一次（非自动）。
- **M1｜三图分离**：实现 DAG Builder / State Store / Resource Graph 三个独立构建器。
- **M2｜异常关联**：实现跨信号的时间对齐与因果链推导（loss spike ↔ 资源 ↔ batch ↔ config）。
- **M3｜查询与发布**：支持"五问"结构化查询与可视化，输出可复现的异常归因报告。

## 十、验收标准

能端到端完成「TrainingGraphFactory」，并在其「核心」（训练全过程 DAG / 状态 / 资源可视化）上给出可复现的量化结果：

- Job DAG / State Graph / Resource Graph 三张图均可独立构建与查询
- 任取一次训练异常，能给出「发生了什么 / 何时 / 哪个资源 / 哪个配置 / 哪个 checkpoint」的完整答案

---

_本 README 的「条目信息」字段来自 `factory-everything-项目总表.md`，架构与流水线为 L3 深化设计稿；可按实际进展修订。_

---

# 附录 A｜骨架、快速开始、CLI 契约、产物

## A.1 目录

```
L3-traininggraphfactory/
├── configs/default.json            所有口径与门槛（代码里不硬编码任何阈值）
├── data/run/events.jsonl           **采集到的事件流**（四路 1585 条，`run` 时写出）
├── src/traininggraphfactory/
│   ├── rng.py                      确定性随机（sha256 派生种子）+ canonical_json
│   ├── schema.py                   对象链：Event → StepSignal → 三张图 → Anomaly → CausalHop → Attribution / TruthRecord
│   ├── config.py                   配置加载 + 校验 + 两类摘要（protocol / version）
│   ├── events.py                   四路异质事件流模拟器 + 事件流的采集与回读
│   ├── align.py                    信标对齐：identity / constant / linear 三种假设同台
│   ├── stats.py                    Theil–Sen 去趋势 + 分层 + 乘性通道的稳健基线
│   ├── jobdag.py                   图一：Job DAG（模板 + 实例化规则）
│   ├── state.py                    图二：Training State Graph（重放 + 三重核对）
│   ├── resource.py                 图三：Resource Graph（每资源自己的稳健 z）
│   ├── detect.py                   分层检出 + episode 聚合 + 显式 NaN 分支
│   ├── recover.py                  签名匹配 → 因果链复原（**一行真值都不读**）
│   ├── verify.py                   第三方校验：P/F/h 探针 + 反事实重放 + α 标定 + 对齐消融
│   ├── lineage.py                  信号血缘 DAG
│   ├── pipeline.py                 编排 + 落盘 + load_result
│   ├── acceptance.py               对抗用例（**不依赖产物**）
│   ├── report.py                   验收表 + 期望核对 + markdown 报告
│   ├── visualize.py                自包含 HTML（零外部资源）
│   └── cli.py                      2 写 + 14 只读子命令
├── tests/                          空 —— 见 A.7
├── main.py                         入口（零安装：把 src/ 塞进 sys.path）
└── out/                            产物（见 A.6）
```

## A.2 形状：第八种 —— **可观察计算图 / 数字孪生**

前七种形状（直线流水线 / 进化闭环 / 搜索闭环 / 扇出-收拢 / 测量型 / 比较型 / 对抗裁判池）
都有一个共同点：**它们在优化或变换某个对象**。本项目没有。

| | 被优化的对象 | 真值判据 |
| --- | --- | --- |
| 其余七种 | 数据集 / 架构 / 适配器 / 偏好集 … | 产物的质量指标 |
| **本形状** | **没有** | **注入的已知量必须被测量回来** |

被生产出来的东西是**一次运行的解释**。于是它有三条别处没有的纪律：

1. **真值与答案必须分开**（`TruthRecord` vs `Attribution`，落成两个文件）。
   复原器一行真值都不读；读真值的是 `verify`，而它是「生成解释的人不给自己打分」里的第三方。
2. **三张图在数据层就分开**（三份产物、三块 SVG）。混在一起就退化成一张杂乱的流程图。
3. **允许设计期望被推翻**。九条声明期望里有三条没有兑现，报告逐条点名。

## A.3 对象链

```
四路异质事件流（job 带 step；metric / resource / io 不带）
        │  align（靠同步信标，不靠时间戳）
        ▼
逐步信号表 StepSignal（120 步 × 18 个信号）
        │
        ├──→ jobdag   → Job DAG（145 实例节点 / 145 边）
        ├──→ state    → State Graph（9 声明状态 / 8 走过）
        ├──→ resource → Resource Graph（7 资源 / 15 步有瓶颈）
        │
        ▼  detect（Theil–Sen 去趋势 + 分层 + 乘性通道）
Anomaly（20 条，跨 10 个信号）
        │  recover（签名匹配 + 主信号贴近性）
        ▼
Attribution（6 起）—— 八层信号上的因果链 + 五问答案
        │  verify（第三方：驱动真实模拟器重放）
        ▼
VerifyResult（P/F/h 探针 + 反事实 + α 曲线 + 对齐消融）
```

**八层信号是固定的**，每一跳都能用 `layer + key` 唯一定位：

| layer | 前缀 | 例子 |
| --- | --- | --- |
| `config` | `cfg:` | `cfg:train.lr@42` |
| `data` | `dat:` | `dat:batch_tokens@55` |
| `io` | `io:` | `io:ckpt_write_s@72` |
| `job` | `job:` | `job:step_time_s@62` |
| `resource` | `res:` | `res:vram_bytes@62` |
| `metric` | `met:` | `met:loss@43` |
| `state` | `st:` | `st:TRAINING->FAILED@97` |
| `checkpoint` | `ckpt:` | `ckpt:ckpt-0096@98` |

## A.4 快速开始

```bash
python main.py run            # 端到端跑一遍，写 out/ 与 data/run/events.jsonl
python main.py visualize      # 另写 out/report.html（自包含，零外部资源）
python main.py acceptance     # 验收表 + 对抗用例
python main.py attribution    # 五问答案
python main.py verify         # 第三方校验
```

零安装、零第三方依赖：`python main.py` 直接把 `src/` 塞进 `sys.path`。

## A.5 CLI 契约

**写命令 2 个**：`run`（含 `--no-capture`）、`visualize`。

**只读命令 14 个**（全部支持 `--json`，且 `--json` 时 stdout 上**只有一段 JSON**）：

```
summary / graphs / job / state / resource / align / anomalies
/ attribution / truth / verify / expectations / acceptance / lineage / events
```

只读命令的取数规则：

- 从 `out/` 的产物**重建**结果对象（`pipeline.load_result`），绝不重跑流水线；
- 产物缺失时在内存里跑一遍 —— **仍然不落盘**，并在输出里明说「本次是内存重跑，未写盘」；
- 唯一的例外是 `events`：它读的是**输入**（`data/run/events.jsonl`）而不是产物。
  文件不在就如实说「还没采集」，不悄悄在内存里造一份。

## A.6 产物

`out/`（15 个文件，`run` 写 14 个 + `visualize` 写 1 个）：

| 文件 | 内容 |
| --- | --- |
| `job_dag.json` | 图一：模板 + 145 实例节点 + 145 边 |
| `state_graph.json` | 图二：声明 9 状态 / 12 边，走过 8 / 10，未走到的如实列出 |
| `resource_graph.json` | 图三：7 资源的压力与逐步瓶颈 |
| `step_signals.csv` | **唯一**一张逐步信号表，其余产物都是它的派生物 |
| `alignment.json` | 三假设对比 + 选中模型 + 步边界（逐步表只存一份，这里不重复） |
| `anomalies.json` | 20 条检出 + 分层基线 + 未判定的步 |
| `attributions.json` | **答案**：6 起归因 |
| `truths.json` | **答案的判据**：6 个注入项 |
| `verify.json` | 第三方校验 |
| `expectations.json` | 九条设计期望 × 实测 |
| `lineage.json` / `lineage_chain.dot` | 信号血缘（185 节点 / 272 边） |
| `version.json` | 内容寻址版本（**唯一含 `created_at` 的文件**） |
| `report.md` | 归因报告 |
| `report.html` | 自包含 HTML（`visualize` 单独写） |

`data/run/events.jsonl`：采集到的四路事件流（1585 条）。它**不进 `out/`** ——
`out/` 装的是「分析的结论」，这里是「分析的输入」，混在一起就分不清
「这份结论对应哪份输入」了。

## A.7 `tests/` 为空

本工作区有一条明确的用户指令：**不做测试只写 pytest 之外的东西** ——
`tests/` 目录留空，验证改放进流水线自身的运行时核对：

- `acceptance.py` 的**对抗用例**（25 条：拒收坏配置 / 拒收坏事件 / 拒收坏状态 / 决定论）；
- `report.py` 的**验收表**（10 项，每条给实测值 / 门槛 / 关系）；
- `verify.py` 的**第三方校验**（探针 + 反事实重放 + 消融）。

这三处都是**运行时真的跑**的，不是被跳过的装饰。

---

# 附录 B｜实测结果

`python main.py run` 一次端到端，版本 `v1-910847f07898`，口径 `bd392f32b4ae`。

## B.0 一句话结论

**两条验收标准都满足**，而且是被**两个独立的判据**同时确认的：
复原器给出 6/6 起归因、五问全对、因果链逐键一致；
第三方（驱动真实模拟器重放）确认 6/6 起归因在去掉对应根因后**真的消失**。

九条设计期望里 **6 条兑现、3 条被推翻**。被推翻的那三条是本次最有价值的产出。

## B.1 两条验收标准

> **① Job DAG / State Graph / Resource Graph 三张图均可独立构建与查询**

| 图 | 规模 | 独立查询入口 | 产物 |
| --- | --- | --- | --- |
| Job DAG | 模板 8 节点 / 9 边 → 实例 **145 节点 / 145 边** | `job` / `job --node` / `job --step` | `job_dag.json` |
| State Graph | 声明 **9 状态 / 12 边**，走过 8 / 10 | `state` / `state --step` | `state_graph.json` |
| Resource Graph | **7 资源**，15/120 步有瓶颈 | `resource` / `resource --step` / `resource --resource` | `resource_graph.json` |

三条「分开」的证据，不是口号：

1. **三个文件**。`load_result` 分别反序列化三张图，任一张缺失就整体返回 `None`。
2. **三种构建输入**。Job DAG 吃 `job` 事件（不需要对齐），State Graph 吃状态转移，
   Resource Graph 吃逐步信号表 —— 三者的**依赖不同**。
3. **Job DAG 拒收异质事件**：非 `job` 源的事件带 `step` 就抛 `GraphError`。
   全项目只有 `job` 允许带步号，这一条在对抗用例里被钉住。

**Job DAG 的实例化规则**（写明了才算「展开」，否则只是一句话）：

| 规则 | 含义 | 实例数 |
| --- | --- | --- |
| `seq` | 同模板相邻实例串联（`train@1 ← train@0`） | 119 |
| `step_match` | 两边在同一步都有实例时相连 | 20 |
| `last_first` | 左端最后一个实例 → 右端第一个实例 | 6 |

任取一步查依赖：`checkpoint@72` ← `train@72`，该节点祖先 **76** 个。
拓扑序前两个是 `load_data@0` → `load_model@0` —— **入度为 0 时按节点 id 排序取下一个**，
否则 `topo_order` 会依赖 `set` 的迭代顺序，两次运行给出不同的序（这条进了对抗用例）。

**State Graph 的两条纪律**：

- 路线：`PREPARING → TRAINING → FAILED → TRAINING → DONE`；
- 没走到的状态 `PAUSED`、没走到的转移 `TRAINING→PAUSED` / `PAUSED→TRAINING` **照样报出来**。
  「声明了但这次没发生」与「声明里根本没有」必须分开 —— 把它画进图里当走过就是谎报覆盖。

步数分布：`TRAINING 117 / DONE 1 / FAILED 1 / PREPARING 1`。

**Resource Graph** 的归一化用**该资源自己的稳健 z**，不是全局 z：

| 资源 | 信号 | 峰值步 | 峰值 | 基线 | 归一化 z | 方向 |
| --- | --- | --- | --- | --- | --- | --- |
| Disk | `disk_mbps` | 73 | 0.48 | 8 | **67.6** | low |
| CPU | `cpu_util` | 55 | 99 | 33.95 | **39.4** | high |
| GPU0 | `gpu0_util` | 98 | 23.08 | 78.84 | **35.3** | low |
| GPU1 | `gpu1_util` | 98 | 23.23 | 74.53 | **34.4** | low |
| VRAM | `vram_bytes` | 63 | 6.674e+09 | 4.453e+09 | **24.9** | high |
| Network | `net_mbps` | 11 | 165.1 | 178.9 | 2.45 | low |
| RAM | `ram_bytes` | 80 | 1.616e+10 | 1.599e+10 | 0.536 | high |

**瓶颈只在压力超过 `k=6` 时报，否则报空**：120 步里只有 **15 步**报了瓶颈，
其余如实报「无显著压力」。报一个「这一步的瓶颈是 RAM，z = 0.5」等于报噪声。

> **② 任取一次训练异常，能给出「发生了什么 / 何时 / 哪个资源 / 哪个配置 / 哪个 checkpoint」的完整答案**

**6/6 起归因，五问全对，因果链逐键一致。**

| 根因 | what | when | resource | config | checkpoint | conf |
| --- | --- | --- | --- | --- | --- | --- |
| `lr_spike@43` | `loss_spike` | 43 | GPU0 | `train.lr`=0.0024 | `ckpt-0036` | 1.00 |
| `bad_batch@55` | `step_time_spike` | 55 | CPU | `data.max_seq_len`=256 | `ckpt-0048` | 1.00 |
| `vram_pressure@62` | `vram_pressure` | 62 | VRAM | `train.micro_batch`=2 | `ckpt-0060` | 1.00 |
| `disk_stall@72` | `disk_stall` | 72 | Disk | `ckpt.every`=12 | `ckpt-0072` | 1.00 |
| `eval_regression@90` | `eval_loss_spike` | 90 | Disk | `data.eval_slice`=eval-v1 | `ckpt-0084` | 0.67 |
| `nan_grad@97` | `loss_nan` | 97 | GPU0 | `train.grad_clip`=1 | `ckpt-0096` | 1.00 |

**因果链（八层信号，从因到果）**：

| 根因 | 链 |
| --- | --- |
| `lr_spike` | `cfg:train.lr@42` → `met:grad_norm@42` → `met:loss@43` |
| `bad_batch` | `dat:batch_tokens@55` → `job:step_time_s@55` → `res:cpu_util@55` |
| `vram_pressure` | `res:vram_bytes@62` → `job:step_time_s@62` → `res:gpu0_util@62` |
| `disk_stall` | `res:disk_mbps@72` → `io:ckpt_write_s@72` → `job:step_time_s@72` |
| `eval_regression` | `met:eval_loss@90` → `res:disk_mbps@90` |
| `nan_grad` | `met:grad_norm@97` → `met:loss@97` → `st:TRAINING->FAILED@97` → `res:gpu0_util@98` → `ckpt:ckpt-0096@98` |

`eval_regression` 的置信度 0.67 是**唯一一个不满分**的 —— 它缺的那一跳是
`job:step_time_s@90`，而这条边是**故意不兑现**的（见 B.6）。缺了它，
签名命中率 2/3 ≥ `min_signature_hit`，仍然成立。

## B.2 对齐：常数偏移 vs 漂移，同台比

只有 `job` 带步号。其余三路必须靠**同步信标**接回来 —— 不靠时间戳，
因为时间戳本身就是被测的对象。1215 个待归属事件，**1212 个接上了步**。

三种假设**同台**，判据是**步归属错误率**而不是拟合优度：

| 来源 | 假设 | 估计偏移 (s) | 估计斜率 | 残差 RMS (s) | 步归属错误率 |
| --- | --- | --- | --- | --- | --- |
| `io` | identity | 0 | 1 | 1.130 | 97.8% |
| `io` | **constant** ← | 1.1316 | 1 | 0.00235 | **0.00%** |
| `io` | linear | 1.1324 | 0.999965 | 0.00211 | 0.00% |
| `metric` | identity | 0 | 1 | 0.371 | 51.0% |
| `metric` | **constant** ← | 0.37121 | 1 | 0.00325 | **0.26%** |
| `metric` | linear | 0.37121 | 0.99999 | 0.00322 | 0.26% |
| `resource` | identity | 0 | 1 | 0.154 | 20.3% |
| `resource` | constant | 0.0071 | 1 | 0.154 | 19.9% |
| `resource` | **linear** ← | −0.21028 | 1.006 | 0.00317 | **0.15%** |

**`resource` 是唯一选中 `linear` 的**：它的 `constant` 残差 0.154s ≫ 容差 0.05s，
**直接出局**，连比都不用比。这就是「常数偏移不够用」的硬证据。

估计精度：

| 来源 | 选中 | 偏移误差 | 漂移误差 |
| --- | --- | --- | --- |
| `io` | constant | 0.00161 s | 0 |
| `metric` | constant | 0.00121 s | 0 |
| `resource` | linear | 0.000275 s | 4.22e-06 |

**`job` 的偏移不是「估出来是 0」，是坐标系选在它上面。** 它的偏移不可辨识，
所以按定义取 0；所有偏移都是**相对 job 的偏移**，绝对值没有意义。

**对齐消融**（真的把 `linear` 拿掉重跑整条流水线）：

| 允许的假设 | 救回归因 | 说明 |
| --- | --- | --- |
| `identity` | **0/6** | `io` 没有一种假设的残差在容差内（最好的是 identity：1.1311s） |
| `identity+constant` | **0/6** | `resource` 没有一种假设的残差在容差内（最好的是 constant：0.1538s） |

于是 `alignment_lift = 100%`：**少了漂移感知，一条归因都复原不出来**。

## B.3 检出：去趋势 + 分层 + 乘性通道

`k=6`，检出 **20 条**，跨 **10 个信号**。

| 信号 | 条数 | 步区间 | 峰值 z（有符号） |
| --- | --- | --- | --- |
| `batch_tokens` | 1 | 55 | 17.6 |
| `ckpt_write_s` | 1 | 72 | 123 |
| `cpu_util` | 2 | 55、62-69 | 39.4 |
| `disk_mbps` | 2 | 72-74、90 | −30 |
| `eval_loss` | 1 | 90 | 13.6 |
| `gpu0_util` | 4 | 55、62-69、72、98 | −13 |
| `grad_norm` | 2 | 42、97 | 45.4 |
| `loss` | 2 | 43-44、97 | 11.5 |
| `step_time_s` | 4 | 55、62-69、72、98 | 93.3 |
| `vram_bytes` | 1 | 62-69 | 24.9 |

z 带符号：`gpu0_util` 是「高才好」，所以它的峰值 z 是**负的** —— 掉下去才是坏事。
丢掉符号就分不出「利用率涨上去」和「掉下来」了。

**为什么必须去趋势**：`loss` 从 4.2 衰减到 0.12（**34×**），全局中位数基线检不出真正的
spike —— 真正的 spike 藏在趋势里。用 Theil–Sen 去趋势（两两斜率的中位数，对 <29% 的
离群点免疫），跨度 ≥ 4 的信号走**对数（乘性）通道**。修好之后 `loss_z` 从「检不出」
变成 11.5、`eval_loss_z` 13.6。

**为什么必须分层**：`ckpt_every=12` 与 `eval_every=10` 每 60 步撞一次，
第 60 步既写 ckpt 又跑 eval，耗时比普通 ckpt 步长 30% —— 在 ckpt 层里是 z≈15 的假阳性。

**分层 vs 不分层（实测）**：

| | 检出条数 |
| --- | --- |
| 分层（默认） | **20** |
| 不分层（消融） | **42** |

多出来的 22 条全是 `step_time_s` / `disk_mbps` 类的假阳性。

**「未判定」与「正常」分开报**：10 个信号各有 1 步因为分层样本不足而**没有判定**
（`ckpt+eval` 混合步只有 1 个样本，不够 `min_stratum_size=4`）。它既不是异常也不是正常，
如实记进 `skipped`。

**NaN 必须显式判异常**：`abs(nan - median) > k*mad` 永远是 False，靠容差比较永远判不出来。
第 97 步的 `loss` 是 NaN，它被显式判为异常，并且**留在逐步信号表里** ——
它是「测到了 nan」，不是「没测到」（后者是 `None`）。

## B.4 第三方校验：生成解释的人不给自己打分

**P / F / h 三个可判分探针**（`alpha_sigma = 1.0`：变化不到一个噪音尺度，与噪音不可区分）：

| 探针 | 问的问题 | 通过率 | 个数 |
| --- | --- | --- | --- |
| **P** | 去掉它，目标信号**要动** | **100%** | 6 |
| **F** | 去掉它，它声明**不影响**的信号必须不动 | **100%** | 6 |
| **h** | 在别处注入等量**不同种类**的诱饵，目标信号必须不动 | **100%** | 6 |

只测 P 是不够的：一个「只要注了就都行」的判据也会全过。所以三个方向都要测。

**反事实反向链**（把某个原因真的从模拟器里去掉，重跑整条流水线）：

| 根因 | 去掉前 | 去掉后 | 同类剩 | 链消失 |
| --- | --- | --- | --- | --- |
| `lr_spike@43` | 6 | 5 | 0 | ✓ |
| `bad_batch@55` | 6 | 5 | 0 | ✓ |
| `vram_pressure@62` | 6 | 5 | 0 | ✓ |
| `disk_stall@72` | 6 | 5 | 0 | ✓ |
| `eval_regression@90` | 6 | 5 | 0 | ✓ |
| `nan_grad@97` | 6 | 5 | 0 | ✓ |

**6/6**：去掉谁，谁的归因就消失；而且**同类剩 0** —— 不是被另一个同类顶替了。

**α 标定曲线**（基线取「幅度压到 1 = 无效果」的那次运行）：

| 根因 | f=0 | 0.05 | 0.1 | 0.2 | 0.35 | 0.5 | 0.75 | 1.0 | f* | 单调 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lr_spike` | 0 | 2.41 | 4.08 | 6.33 | 8.46 | 9.89 | 11.5 | 12.7 | **0.05** | 是 |
| `bad_batch` | 0 | 9.93 | 16.4 | 24.8 | 32.9 | 38.5 | 45.3 | 50.5 | **0.05** | 是 |
| `vram_pressure` | 0 | 1.25 | 2.49 | 4.99 | 8.73 | 12.5 | 18.7 | 24.9 | **0.05** | 是 |
| `disk_stall` | 0 | 0.932 | 1.81 | 3.48 | 5.93 | 8.5 | 13.7 | 23.3 | **0.10** | 是 |
| `eval_regression` | 0 | 1.01 | 1.99 | 3.81 | 6.31 | 8.56 | 11.9 | 14.8 | **0.05** | 是 |
| `nan_grad` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | —（开关式） | 是 |

**`nan_grad` 的曲线必然退化。** 它是开关式的：梯度变成 NaN 这件事没有「幅度小一点」的
中间态。一条测不出中间态的曲线不是流水线的缺陷，是它诚实报告了「这个原因不可调幅」。

## B.5 设计期望 vs 实测 —— **3 条被推翻**

| 期望 | 声明 | 实测 | 判定 | 依据 |
| --- | --- | --- | --- | --- |
| `alignment_is_necessary` | 是 | 成立 | ✓ 兑现 | 对齐消融：常数偏移只剩 0/6 条归因，lift=100.0% |
| `constant_clock_offset_suffices` | 是 | **不成立** | ✗ **被推翻** | 只用 identity+constant：0/6 条归因（`resource` 残差 0.1538s 直接出局） |
| `single_global_baseline_suffices` | 是 | **不成立** | ✗ **被推翻** | 不分层检出 42 条 vs 分层 20 条 |
| `declared_signatures_all_materialize` | 是 | **不成立** | ✗ **被推翻** | 未兑现的签名标签 1 个：`eval_regression:step_time_z` |
| `declared_causal_model_attributes_uniquely` | 是 | 成立 | ✓ 兑现 | 有歧义的归因 0 起 |
| `all_injected_anomalies_detected` | 是 | 成立 | ✓ 兑现 | 召回 6/6 = 100.0% |
| `no_false_positive_anomalies` | 是 | 成立 | ✓ 兑现 | 未被任何归因覆盖的检出 0 条 |
| `three_graphs_separately_queryable` | 是 | 成立 | ✓ 兑现 | job 145 节点 / state 8 状态 / resource 7 资源，三份产物三个文件 |
| `recovery_lands_on_last_clean_checkpoint` | 是 | 成立 | ✓ 兑现 | checkpoint 全对 6/6 |

**被推翻的三条，各自值多少钱**：

- **`constant_clock_offset_suffices`**：这条是**整个对齐模块存在的理由**。
  声明说「常数偏移就够了」，实测说 `resource` 的 constant 残差 0.1538s 是容差的 3 倍，
  连门都进不去。删掉 `linear` 分支，本项目一条归因都复原不出来。
- **`single_global_baseline_suffices`**：声明说「一条全局基线就够了」，实测说会多出 22 条
  假阳性，而且会把 `vram_pressure` 的链污染成 `@60`（混合步）。分层不是精度优化，是正确性前提。
- **`declared_signatures_all_materialize`**：这条**是有意不兑现的**。
  `eval_regression` 的因果模型声明了 `step_time_z`（「评测变慢 → 步变慢」），
  但传播规则里**故意不实现这条边** —— 因为真实的评测回归不一定拖慢训练步。
  于是「声明的信念」与「兑现的物理」之间裂开一条缝，而 `expectations.json`
  把这条缝**指了出来**。这是全项目唯一一条「设计文档写错了、而流水线测出来了」的记录。

## B.6 决定论与只读纯度

**两次运行（相同输入）**：`out/` 15 个文件里 14 个**逐位相同**，
唯一不同的是 `version.json` 的 `created_at` —— 这是设计允许的唯一非决定论字段。

为此报告正文里**不印 `created_at`**：印了它，两份相同输入的报告就不逐位相同了。

**只读命令**：14 个命令 × 25 种调用形态（含 `--node` / `--step` / `--signal` / `--kind` /
`--dot` / `--source` / `--limit`）全部返回**纯 JSON**（`json.loads` 直接成功），
跑完之后 `out/` 与 `data/` 的 md5 全部不变。

**对抗用例 25/25**，全部**不依赖产物**（只吃配置与合成事件）：

| 类别 | 条数 | 例子 |
| --- | --- | --- |
| 拒收坏配置 | 6 | `root_causes` 带 `config_key`；两个 `job` 源；`job` 源带非零偏移 |
| 拒收坏事件 | 1 | Job DAG 收到「非 job 源却带步号」的事件 |
| 拒收坏状态 | 3 | 未声明的状态名 / 未声明的转移边 / 起点对不上 |
| 拒收坏标签 | 4 | `evidence_hop` 缺参数；`hop_key` 用了未声明的层 |
| 决定论 | 4 | 两次 `simulate` 逐位相同；打乱事件顺序后 DAG 相同；`topo_order` 稳定 |
| 事件流形状 | 2 | 不带步号的源强制 `step=None`；四路都有信标 |
| NaN 与分层 | 3 | NaN 留在表里；混合步单独成层；样本不足时不判定 |
| 答案不抄答案 | 1 | 归因里不出现真值项编号 |
| 分层有效性 | 1 | 不分层会多出成批假阳性 |

## B.7 验收

| 验收项 | 实测 | 门槛 | 关系 |
| --- | --- | --- | --- |
| `min_anomaly_recall` | 1 | 1 | >= |
| `max_anomaly_false_positives` | 0 | 0 | <= |
| `min_five_answer_rate` | 1 | 0.8 | >= |
| `min_chain_exact_rate` | 1 | 0.6 | >= |
| `max_offset_error_s` | 0.0016105 | 0.05 | <= |
| `max_drift_error_s_per_s` | 4.2209e-06 | 0.0005 | <= |
| `min_alignment_lift` | 1 | 0.2 | >= |
| `min_state_coverage` | 8 | 6 | >= |
| `min_resource_coverage` | 7 | 7 | >= |
| `min_job_nodes` | 145 | 20 | >= |

**10/10 通过。**

**血缘**：185 节点 / 272 边。两条别处没有的不变量 ——

1. `attrib:<kind>@<step>` 的父节点**恰好是**它 `covered_anomalies` 里那几个检出
   （一个不多、一个不少）；
2. `hop:<layer>:<key>` 的父节点**恰好是**链上前一跳 —— 从任意一跳往上走，走到根就是第一因。

---

# 附录 C｜坑（161–180）

编号**跨项目连续**：`L3-adapterfactory` 的附录 C 止于 **160**，本项目占 **161–180**
（**下一个项目从 181 开始**）。

> **为什么不是 126。** 本项目起草时，工作区里附录 C 最靠前的编号是
> `L3-finetunefactory` 的 108，于是按「接着 108」写成了 126 —— 但同一天先交付的
> `L3-adapterfactory` 已经占了 **126–160**（它自己是让了 `L4-preferencefactory` 的
> 109–125 之后后移 17 位落到 126 的）。两个项目各自都算对了，合起来才撞车。
> 编号连续这件事的意义**恰恰是「坑号全局唯一」**，一旦允许两个 130，这条约定就
> 只剩下一个形式。所以本项目整体后移 **35** 位到 161 起。
>
> 这是**同一个坑在一周内第二次发生**（上一次是 PreferenceFactory 与 AdapterFactory）。
> 教训不在「要算对」，而在：**「连续」是一个跨进程的共享资源，而每个进程只能看见
> 自己开始时的快照。** 唯一能自动化的是**交付前重新扫一遍全工作区的最大编号** ——
> 而不是起草时算一次就假定它成立。

### 161. 三种图混在一起，就退化成一张杂乱的流程图

**症状**：第一版把 Job / State / Resource 画在一张 SVG 里。看图的人第一件事是去找
「哪个框属于哪张图」。

**原因**：三张图的**节点类型不同**（作业步骤 / 状态 / 资源），但它们共享「step」这一个维度，
所以画在一起「看起来是合理的」。

**修法**：**在数据层就分开** —— 三个构建器、三个产物文件、三块 SVG。
`jobdag` / `state` / `resource` 三者的构建输入也刻意不同（job 事件 / 状态转移 / 逐步信号表）。

### 162. 不带步号的源必须**强制** `step=None`

**症状**：`metric` 采样在生成时顺手把 `step` 填上了，于是对齐变成了一句空话 ——
信号「天生就知道自己在第几步」。

**修法**：`emit()` 里对不带步的源**强制**置 `None`，并在对抗用例里钉住
（「越界事件 0 条」）。全项目只有 `job` 允许带 step，`jobdag` 会拒收其余。

### 163. `job` 的偏移不可辨识，按定义取 0

**症状**：一度想让 `job` 也参与偏移估计，于是「谁是对的」变成了循环论证。

**修法**：**坐标系钉在 `job` 上**。它的偏移不是「估出来是 0」，是定义。
所有偏移都是**相对 job 的偏移**，绝对值没有意义。报告里明写这一点。

### 164. 漂移的线性项必须对**真时间**，不能用 `t_local` 当自变量

**症状**：用 `t_local` 拟合 `t_local = a + b·t_local`，得到 `b = 1` 恒成立。

**修法**：模型是 `t_local = t_true + offset + drift·t_true`，自变量是**真时间**。
反解用 `t_true = (t_local − offset) / (1 + drift)`。

### 165. 对齐靠**信标序号**，不靠时间戳

**症状**：想按时间戳找「最近的一对信标」，结果在漂移下配错。

**修法**：四路各自发出的第 k 个信标对应同一个真时刻，所以**按序号配对**。
序号是离散的、不会因为漂移而错位。

### 166. 步边界必须插值，不能假设步长均匀

**症状**：按 `step × 平均步长` 算边界，异常步（recover 的步长是正常的 10 倍）
整段错位。

**修法**：用 `bisect` 查真实的 `[t_true_begin, t_true_end)` 表。步长本来就会被异常改变，
均匀假设在最需要它准的时候最不准。

### 167. 判据是**步归属错误率**，不是拟合优度

**症状**：一度按残差 RMS 挑模型，于是 `linear` 因为多一个参数而永远赢一点点。

**修法**：残差只用来**淘汰**（`identity` 的残差等于偏移本身，直接出局），
在剩下的里按**步归属错误率**排序；打平则取更简单的模型 —— `linear` 必须**挣得**它的第二个参数。

### 168. `loss` 的 34× 衰减让全局基线完全失效

**症状**：`loss` 从 4.2 衰减到 0.12，真正的 spike 藏在趋势里，`z` 永远不到 6。

**修法**：Theil–Sen 去趋势（两两斜率的中位数，对 <29% 的离群点免疫）+
跨度 ≥ `detrend_span` (4.0) 的信号走**对数通道**。修好之后 `loss_z` = 11.5。

### 169. `ckpt+eval` 混合步必须单独成层

**症状**：第 60 步既写 ckpt 又跑 eval，耗时比普通 ckpt 步长 30%，
在 `ckpt` 层里是 **z≈15 的假阳性**，还顺手把 `vram_pressure` 的链污染成 `@60`。

**修法**：`step_kind()` 返回四类（`train` / `ckpt` / `eval` / `ckpt+eval`）。
判据来自**事件本身**（这一步真的写了 ckpt / 真的跑了 eval），不是配置里的周期数字 ——
周期可以被改，「真的发生了」不会。

### 170. 分层样本不足时**不判定**，不拿全量基线兜底

**症状**：混合步只有 1 个样本，不够 `min_stratum_size=4`。于是「退回全量基线」
看起来很自然 —— 但那是「不分层」那个错误的翻版。

**修法**：`stratum_fallback: "none"`。没有基线就是**未判定**，记进 `skipped`。
「未判定」与「正常」必须分开报，否则「未判定」会被读成「没问题」。

### 171. NaN 必须显式判异常，且必须留在表里

**症状**：`abs(nan - median) > k*mad` 永远是 `False` —— 靠容差比较**永远**判不出 NaN。

**修法**：`detect` 里给 NaN 一个显式分支（一定异常，且不进 z 通道）；
步内聚合**只要有一条 nan，这一步就是 nan**（把 nan 平均掉比没有数更糟）；
CSV 里 `nan` 与空串是两件事 —— 前者是「测到了 nan」，后者是「没测到」。

### 172. 证据步的约定是「首次出现」，不是峰值

**症状**：真值链用峰值步，复原链用首次步，两边在 `disk_z_drop` 上于 72 / 73 之间摆动。

**原因**：一次持续 3 步的**等比例**跌幅里，「哪一步是峰」是**平局**。
让两边各自猜平局，等于让链的比对变成比运气。

**修法**：统一成「首次出现」（`_evidence_steps`）。平局一律按声明顺序/首次出现裁决。

### 173. 逐步报异常会让召回虚高

**症状**：一次持续 6 步的 lr spike 被报成 6 条异常。

**修法**：**episode 聚合** —— 连续被标记的步聚成一段。20 条检出对应 6 起根因，
而不是 20 起。

### 174. `root_causes` 里不许带 `config_key` / `resource`

**症状**：注入项里写了 `config_key: "train.lr"`，于是复原器「复原」出来的
`config_key` 就是从注入项抄的。

**修法**：`validate_config` **显式拒收**这两个字段 —— 它们是 `causal_model` 的职责，
不是注入实例的职责。声明的因果模型与注入实例必须分开。

### 175. `throughput` 噪声 11.6%，当不了稳健 z 的判据

**症状**：`vram_pressure` 的签名里写了 `throughput_z_drop`，但它从来检不出 ——
噪声 11.6%，要 z=6 得跌 70%。

**修法**：换成 `gpu_util_z_drop`（利用率掉下去是显式的、低噪声的信号）。
**一条永远检不出的签名不是「严格」，是「不存在」。**

### 176. `bad_batch` 的资源是 CPU，不是 GPU0

**症状**：第一版把 `bad_batch` 归到 GPU0，因为它最终表现为 GPU 空转。

**修法**：坏批次是**宿主侧**打爆的（`cpu_util` 99 / 基线 34），GPU 空转是**后果**。
资源字段填的是**因所在的资源**，不是**果所在的资源**。

### 177. 幅度必须是**物理参数**，否则 α 标定曲线扫不动

**症状**：`magnitude` 只写进 `config_value`，异常幅度是**硬编码**的 ——
于是扫幅度完全不起作用，曲线全零。

**修法**：所有乘数按 `sev = ln(mag) / ln(ref_mag)` 的 `_scale(base, sev) = 1 + (base-1)·sev` 缩放。
`_REF_MAG` 表恰等于默认幅度，于是**默认运行逐位不变**，但幅度扫描在物理上自洽。

### 178. α 曲线的基线要取「幅度压到 1」的运行

**症状**：拿原运行当基线，曲线从「满效果」走到 0 —— 方向是反的。

**原因**：要回答的问题是「幅度小到什么程度就**不再**是原因」，
所以基线必须是**无效果**那一端。

### 179. 自环会让层次松弛发散

**症状**：Job 模板的 SVG 宽 2980px，模板被推到第 14 列。

**原因**：`train → train`（相邻实例串联）是**自环**，
松弛里 `level[train] < level[train] + 1` 每轮都为真，9 轮之后涨到 9。

**修法**：层次计算里 `if a == b: continue`。自环单独画在节点上方。

### 180. 自己拼层前缀会让「所有签名都未兑现」

**症状**：`declared_signatures_all_materialize` 报「16 个签名标签未兑现」——
看起来像个惊天动地的发现。

**原因**：层名是 `metric`，键前缀是 `met`。自己拼 `f"{layer}:{name}@"`
得到一个**永远匹配不上**的字符串。

**修法**：比对必须走 `hop_key(layer, name)`。三个特殊标签（`config_change` /
`state_failed` / `ckpt_restore`）的键不是信号名，而是「当时的那个东西」的名字，
要从这条真值自己身上取（`cfg:train.lr` / `st:TRAINING->FAILED` / `ckpt:ckpt-0096`）。

**教训**：**一个「全都失败」的结论，第一嫌疑是比对代码，不是被测对象。**
真要有 16 条签名同时不兑现，那 6 起归因不可能全对。
