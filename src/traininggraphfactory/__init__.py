"""TrainingGraphFactory —— 训练运行的可观察计算图（数字孪生）。

核心命题：**这个项目不是 Training UI。** 它要把一次训练跑成**一张可追溯的因果图**，
使得任何一次异常都能回答五个问题：

> 发生了什么？在什么时候？落在哪个资源上？当时是哪份配置？影响到了哪个 checkpoint？

为此三张图必须在**数据层就分开**（混在一起就会退化成一张杂乱的流程图）：

| 图 | 回答的问题 | 谁构建 |
| --- | --- | --- |
| Job DAG | 这次运行有哪些作业、谁依赖谁 | `jobdag.py` |
| State Graph | 运行时经历了哪些状态、怎么迁移的 | `state.py` |
| Resource Graph | 每一步压在哪个资源上、瓶颈在哪 | `resource.py` |

因果链则**跨层**：`config → data → io → job → resource → metric → state → checkpoint`。

外部依赖：**零**。重型接入点（torch / matplotlib）一律写成抛 `NotImplementedError`
的扩展点，不进核心链路。
"""

from __future__ import annotations

__version__ = "0.1.0"
