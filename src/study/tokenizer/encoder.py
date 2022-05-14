"""Byte pair encoding utilities"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from .. import cfg

_MAX_RANK = 999999999


def ranks():
    """get ranks"""
    with open(cfg.TOKEN_BPE, 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [
        tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]
    ]
    return dict(zip(bpe_merges, range(len(bpe_merges))))


BPE_RANKS = ranks()


@dataclass
class Pair:
    """pair structure"""
    prefix: str
    suffix: str
    index: int
    p_index: Optional[int]  # previous pair index
    n_index: Optional[int]  # next pair index
    rank: int

    @property
    def mergable(self) -> bool:
        """if this pair can be merged"""
        return self.rank < _MAX_RANK

    @property
    def merged(self) -> str:
        """merged string"""
        return f"{self.prefix}{self.suffix}"

    @classmethod
    def build(
        cls,
        prefix: str,
        suffix: str,
        index: int,
        previous: int,
        latter: int,
    ) -> Pair:
        """build a pair instance"""
        rank = BPE_RANKS.get((prefix, suffix), _MAX_RANK)
        return cls(prefix, suffix, index, previous, latter, rank)


@dataclass
class Tokens:
    """a group of pairs from several tokens"""
    pairs: dict[int, Pair]

    @property
    def indices(self) -> list[int]:
        """indices"""
        return list(self.pairs)

    @property
    def ordered_pairs(self) -> list[Pair]:
        """char-sequence ordered indices"""
        return sorted(self.pairs.values(), key=lambda pair: pair.index)

    @property
    def outcome(self) -> list[str]:
        """outcome of final result"""
        return ([self.ordered_pairs[0].prefix] +
                [p.suffix for p in self.ordered_pairs])

    @property
    def top_ranked_pair(self) -> Pair:
        """ranked index"""
        return min(self.pairs.values(), key=lambda pair: pair.rank)

    @property
    def mergable(self) -> bool:
        """whether or not can be furthur merged"""
        return self.top_ranked_pair.mergable


class Tokenizer:
    """tokenize and encode strings"""

    @staticmethod
    def _utf8_encode(ids: tuple[int, ...]) -> tuple[str, ...]:
        with open(cfg.TOKEN_ALPHABET, 'r', encoding='utf8') as f:
            dc = json.load(f)
        return tuple(map(lambda i: dc[str(i)], ids))

    @classmethod
    def _str_to_pairs(cls, txt: str) -> dict[int, Pair]:
        """a word string to pairs"""

        def _get_index(idx: int, length: int) -> Optional[int]:
            """
            previous / next pair index should not be negative or greater than
            or equal to the length
            """
            if idx < 0 or idx >= length - 1:
                return None
            return idx

        seq = []
        chars = cls._utf8_encode(tuple(txt.encode('utf8')))
        for i in range(len(chars) - 1):
            seq.append(
                Pair.build(
                    chars[i],
                    chars[i + 1],
                    i,
                    _get_index(i - 1, len(chars)),
                    _get_index(i + 1, len(chars)),
                ))
        return {p.index: p for p in seq}

    @staticmethod
    def _merge_one(pairs: Tokens) -> Tokens:
        """merge a pair"""
        top_pair = pairs.top_ranked_pair
        pairs_ = {
            k: v
            for k, v in pairs.pairs.items()
            if k not in [top_pair.index, top_pair.p_index, top_pair.n_index]
        }
        n_index = top_pair.n_index
        if top_pair.p_index is not None:
            p = pairs.pairs.get(top_pair.p_index)
            p_pair = Pair.build(
                p.prefix,
                top_pair.merged,
                p.index,
                p.p_index,
                top_pair.n_index,
            )
            pairs_ = {**pairs_, p_pair.index: p_pair}
        if n_index is not None:
            n = pairs.pairs.get(n_index)
            n_pair = Pair.build(
                top_pair.merged,
                n.suffix,
                n.index,
                top_pair.p_index,
                n.n_index,
            )
            pairs_ = {**pairs_, n_pair.index: n_pair}
        return Tokens(pairs_)

    @staticmethod
    def _merge_all(pairs: Tokens) -> Tokens:
        """merge all pairs possible"""
        while True:
            if not pairs.mergable:
                return pairs
            pairs = Tokenizer._merge_one(pairs)

    @classmethod
    def str_to_tokens(cls, s: str) -> list[str]:
        """string to tokens"""
        dc = cls._str_to_pairs(s)
        token = Tokens(dc)
        return cls._merge_all(token).outcome

    @classmethod
    def encode(cls, s: str) -> list[int]:
        """encoding"""
        tokens = cls.str_to_tokens(s)
        with open(cfg.TOKEN_ID, 'r', encoding='utf8') as f:
            m = json.load(f)
        return list(map(lambda t: m.get(t, 50256), tokens))

    @classmethod
    def decode(cls, ids: list[int]) -> str:
        """decode"""
        with open(cfg.TOKEN_ID, 'r', encoding='utf8') as f:
            m = json.load(f)
        dc = {v: k for k, v in m.items()}
        seq = [dc[i] for i in ids]
        # put back whitespace
        return ''.join(seq).replace('Ġ', ' ')
