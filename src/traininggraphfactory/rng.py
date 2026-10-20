"""确定性随机与稳定哈希。

**禁用内置 `hash()`**：`PYTHONHASHSEED` 会让它跨进程不稳定，直接破坏「同一份配置
跑两次得到逐位相同的产物」这条底线。所有指纹一律走 `sha256`。

本模块是整个项目的**可复现性地基**：事件流的抖动、采样器的相位、时钟偏移的估计
残差全部从这里出去。四条纪律：

1. 任何要进指纹的数据，先过 `canonical_json`（键排序 + 分隔符固定 + 非 ASCII 原样），
   否则 dict 插入顺序会悄悄改变指纹。
2. `Rng` **不直接用** `random.Random.gauss` / `.shuffle` / `.sample` —— 这些方法的
   实现细节（尤其是 `gauss` 的缓存值、`sample` 的选取算法）在不同 CPython 小版本里
   改过。这里只用 `random.Random.random()`（MT19937，**有跨版本稳定性保证**），
   正态分布与洗牌都在它上面自己实现。
3. `derive_seed` 用 `\\x1f` 作分隔符（不可能出现在 id / 名字里），避免
   `("ab", "c")` 与 `("a", "bc")` 撞到同一个种子。
4. `short_digest` 只用于展示；**任何比较都必须用全长 digest**。
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Iterable, Sequence

__all__ = [
    "canonical_json",
    "sha256_text",
    "sha256_bytes",
    "stable_hash64",
    "short_digest",
    "digest_of",
    "derive_seed",
    "join_digest",
    "Rng",
]

_SEP = "\x1f"


def canonical_json(obj: Any) -> str:
    """把任意可序列化对象压成**唯一**字符串表示。

    排序键、固定分隔符、`ensure_ascii=False`、不允许 NaN。凡是进指纹的东西都必须
    先过这里 —— 这是「同一份输入必然得到同一个指纹」的唯一保证。
    """
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=_fallback,
    )


def _fallback(obj: Any) -> Any:
    # tuple 会退化成 list，于是 (1,2) 与 [1,2] 指纹相同 —— 这正是我们想要的：
    # 指纹只关心「值」，不关心容器类型。set 必须排序，否则集合的迭代顺序会进指纹。
    if isinstance(obj, (tuple, set, frozenset)):
        return sorted(obj) if isinstance(obj, (set, frozenset)) else list(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"不可序列化，无法进指纹：{type(obj)!r}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_of(obj: Any, *, length: int = 12) -> str:
    """对象的短指纹（展示用）。比较请用 `sha256_text(canonical_json(obj))` 全长。"""
    return sha256_text(canonical_json(obj))[:length]


def stable_hash64(*parts: Any) -> int:
    """把任意成分折成 64 位整数（供 `random.Random(seed)` 使用）。

    成分先过 `str()`：调用点很自然会传 `(来源 id, 序号)` 这种混合类型，
    而 `"\\x1f".join` 遇到 int 会抛 `TypeError` —— 报错的位置离真正的原因很远。
    """
    return int(sha256_text(_SEP.join(str(p) for p in parts))[:16], 16)


def short_digest(full: str, length: int = 12) -> str:
    return full[:length]


def derive_seed(*parts: Any) -> int:
    """从若干成分派生一个稳定种子。

    分隔符用 `\\x1f` 而不是 `":"`：一旦分隔符可能出现在成分里，
    `("a:b","c")` 与 `("a","b:c")` 就会撞种子，而撞种子的后果是「两个不同来源
    用了同一串随机数」—— 那种 bug 在统计上表现为「两路信号异常同步」，极难归因。
    """
    return stable_hash64(*parts)


def join_digest(parts: Iterable[str]) -> str:
    """把若干 digest 合成一个（用于「版本摘要 = 全部输入摘要的函数」）。"""
    return sha256_text(_SEP.join(sorted(parts)))


class Rng:
    """确定性随机源。**只用** `random.Random.random()`（MT19937，跨版本稳定）。

    需要正态分布时走自己实现的 Box–Muller：一次取两个均匀量、产出两个正态量，
    第二个缓存起来。这比调用 `random.gauss` 安全 —— 后者的缓存策略是实现细节。
    """

    __slots__ = ("_r", "_spare")

    def __init__(self, seed: int | str) -> None:
        self._r = random.Random(seed if isinstance(seed, int) else stable_hash64(seed))
        self._spare: float | None = None

    # ---- 基础

    def random(self) -> float:
        return self._r.random()

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self._r.random()

    def randint(self, lo: int, hi: int) -> int:
        """闭区间 [lo, hi]。"""
        return lo + int(self._r.random() * (hi - lo + 1))

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        if self._spare is not None:
            z = self._spare
            self._spare = None
            return mu + sigma * z
        u1 = self._r.random()
        u2 = self._r.random()
        # u1 不能为 0，否则 log 发散。MT19937 能取到 0，这里必须挡。
        if u1 <= 0.0:
            u1 = 1e-12
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        self._spare = r * math.sin(theta)
        return mu + sigma * r * math.cos(theta)

    def lognormal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return math.exp(self.normal(mu, sigma))

    def bernoulli(self, p: float) -> bool:
        return self._r.random() < p

    # ---- 序列操作（自己实现，避开 `random.shuffle` / `random.sample`）

    def pick(self, seq: Sequence[Any]) -> Any:
        if not seq:
            raise ValueError("pick 不能作用于空序列")
        return seq[int(self._r.random() * len(seq))]

    def shuffled(self, seq: Sequence[Any]) -> list[Any]:
        """Fisher–Yates。**返回新列表**，不改入参 —— 调用点经常要保留原顺序做对照。"""
        out = list(seq)
        for i in range(len(out) - 1, 0, -1):
            j = int(self._r.random() * (i + 1))
            out[i], out[j] = out[j], out[i]
        return out

    def sample(self, seq: Sequence[Any], k: int) -> list[Any]:
        """无放回抽样。k >= len(seq) 时返回全部的一个洗牌（不是报错 —— 对抗用例
        常常比可用样本还大，那种情况下「全用上」比「炸掉」有用）。"""
        if k < 0:
            raise ValueError(f"sample 的 k 不能为负，得到 {k}")
        return self.shuffled(seq)[:k]

    # ---- 派生

    def fork(self, *parts: Any) -> "Rng":
        """从当前源派生一个**独立**子源。

        每个来源 / 每个异常各自 fork 一条，于是「多注入一个异常」不会改变已有
        异常的抖动 —— 否则往配置里加一条根因，所有历史事件都会变。
        """
        return Rng(derive_seed(*parts, self._r.random()))
