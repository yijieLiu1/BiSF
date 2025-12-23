#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detection metrics tracker for poisoning detection.
"""
from typing import Dict, List, Optional, Sequence


class DetectionMetricsTracker:
    """Track detection metrics across rounds and percentile buckets."""

    def __init__(self, total_rounds: int, methods: Sequence[str]) -> None:
        """Initialize tracker with total rounds and method names.

        Args:
            total_rounds: Total federated rounds.
            methods: Detection method names.
        """
        self.total_rounds = max(0, int(total_rounds))
        self.methods = list(methods)
        self._bucket_edges = self._build_bucket_edges(self.total_rounds)
        self._stats: Dict[str, List[Dict[str, int]]] = {
            method: [self._empty_stat() for _ in range(4)] for method in self.methods
        }

    def update(
        self,
        method: str,
        round_idx: int,
        suspects: Optional[Sequence[int]],
        poisoned_ids: Optional[Sequence[int]],
        active_ids: Optional[Sequence[int]],
    ) -> None:
        """Update stats for one round.

        Args:
            method: Detection method name.
            round_idx: 1-based round index.
            suspects: Detected suspect DO ids.
            poisoned_ids: Ground-truth poisoned DO ids.
            active_ids: Active DO ids for this round.
        """
        if method not in self._stats:
            return
        bucket = self._bucket_index(round_idx)
        suspect_set = set(suspects or [])
        poisoned_set = set(poisoned_ids or [])
        active_list = list(active_ids or [])
        if not active_list:
            return

        tp = tn = fp = fn = 0
        for do_id in active_list:
            pred = do_id in suspect_set
            actual = do_id in poisoned_set
            if pred and actual:
                tp += 1
            elif pred and not actual:
                fp += 1
            elif not pred and actual:
                fn += 1
            else:
                tn += 1
        stat = self._stats[method][bucket]
        stat["tp"] += tp
        stat["tn"] += tn
        stat["fp"] += fp
        stat["fn"] += fn
        stat["rounds"] += 1
        poison_present = len(poisoned_set) > 0
        if poison_present:
            stat["poison_rounds"] += 1
        if tp > 0:
            stat["detect_rounds"] += 1
        if suspect_set == poisoned_set:
            stat["exact_match_rounds"] += 1
        if poison_present and tp == 0:
            stat["miss_rounds"] += 1

    def format_summary(self, label: str) -> List[str]:
        """Return formatted summary lines for all methods.

        Args:
            label: Summary label.

        Returns:
            Lines to print or log.
        """
        lines: List[str] = []
        ranges = self._bucket_ranges()
        lines.append(f"Detection metrics ({label}):")
        lines.append(
            "Definitions: acc=(TP+TN)/total, prec=TP/(TP+FP), rec=TP/(TP+FN), "
            "f1=2*prec*rec/(prec+rec); det_rate=有投毒轮次中至少抓到1个投毒DO的比例; "
            "exact_match=检测集合与真实投毒集合完全一致的比例; "
            "miss_rate=有投毒轮次中完全未检测到投毒DO的比例."
        )
        for method in self.methods:
            lines.append(f"[{method}]")
            for idx, (start, end) in enumerate(ranges):
                stat = self._stats[method][idx]
                lines.append(f"  {self._range_label(start, end)}: {self._format_metrics(stat)}")
                lines.append(f"    {self._format_extra(stat)}")
            overall = self._sum_stats(self._stats[method])
            lines.append(f"  Overall: {self._format_metrics(overall)}")
            lines.append(f"    {self._format_extra(overall)}")
        return lines

    def _build_bucket_edges(self, total_rounds: int) -> List[int]:
        if total_rounds <= 0:
            return [0, 0, 0, 0]
        q1 = max(1, (total_rounds * 1 + 3) // 4)
        q2 = max(q1, (total_rounds * 2 + 3) // 4)
        q3 = max(q2, (total_rounds * 3 + 3) // 4)
        q4 = max(q3, total_rounds)
        return [q1, q2, q3, q4]

    def _bucket_index(self, round_idx: int) -> int:
        q1, q2, q3, _ = self._bucket_edges
        if round_idx <= q1:
            return 0
        if round_idx <= q2:
            return 1
        if round_idx <= q3:
            return 2
        return 3

    def _bucket_ranges(self) -> List[tuple]:
        q1, q2, q3, q4 = self._bucket_edges
        return [
            (1, q1),
            (q1 + 1, q2),
            (q2 + 1, q3),
            (q3 + 1, q4),
        ]

    @staticmethod
    def _empty_stat() -> Dict[str, int]:
        return {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "rounds": 0,
            "poison_rounds": 0,
            "detect_rounds": 0,
            "exact_match_rounds": 0,
            "miss_rounds": 0,
        }

    @staticmethod
    def _sum_stats(stats_list: List[Dict[str, int]]) -> Dict[str, int]:
        total = {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "rounds": 0,
            "poison_rounds": 0,
            "detect_rounds": 0,
            "exact_match_rounds": 0,
            "miss_rounds": 0,
        }
        for stat in stats_list:
            total["tp"] += stat["tp"]
            total["tn"] += stat["tn"]
            total["fp"] += stat["fp"]
            total["fn"] += stat["fn"]
            total["rounds"] += stat["rounds"]
            total["poison_rounds"] += stat["poison_rounds"]
            total["detect_rounds"] += stat["detect_rounds"]
            total["exact_match_rounds"] += stat["exact_match_rounds"]
            total["miss_rounds"] += stat["miss_rounds"]
        return total

    @staticmethod
    def _format_metrics(stat: Dict[str, int]) -> str:
        tp = stat["tp"]
        tn = stat["tn"]
        fp = stat["fp"]
        fn = stat["fn"]
        total = tp + tn + fp + fn
        if total <= 0:
            return "acc=n/a, prec=n/a, rec=n/a, f1=n/a"
        acc = (tp + tn) / total
        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = None
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        prec_str = "n/a" if prec is None else f"{prec * 100:.2f}%"
        rec_str = "n/a" if rec is None else f"{rec * 100:.2f}%"
        f1_str = "n/a" if f1 is None else f"{f1 * 100:.2f}%"
        return f"acc={acc * 100:.2f}%, prec={prec_str}, rec={rec_str}, f1={f1_str}"

    @staticmethod
    def _format_extra(stat: Dict[str, int]) -> str:
        rounds = stat.get("rounds", 0)
        poison_rounds = stat.get("poison_rounds", 0)
        detect_rounds = stat.get("detect_rounds", 0)
        exact_rounds = stat.get("exact_match_rounds", 0)
        miss_rounds = stat.get("miss_rounds", 0)
        det_rate = detect_rounds / poison_rounds if poison_rounds > 0 else None
        exact_rate = exact_rounds / rounds if rounds > 0 else None
        miss_rate = miss_rounds / poison_rounds if poison_rounds > 0 else None
        det_str = "n/a" if det_rate is None else f"{det_rate * 100:.2f}%"
        exact_str = "n/a" if exact_rate is None else f"{exact_rate * 100:.2f}%"
        miss_str = "n/a" if miss_rate is None else f"{miss_rate * 100:.2f}%"
        return (
            "confusion: "
            f"TP={stat.get('tp', 0)}, TN={stat.get('tn', 0)}, "
            f"FP={stat.get('fp', 0)}, FN={stat.get('fn', 0)}; "
            f"det_rate={det_str}, exact_match={exact_str}, miss_rate={miss_str}"
        )

    @staticmethod
    def _range_label(start: int, end: int) -> str:
        if start > end:
            return "Rounds n/a"
        return f"Rounds {start}-{end}"
