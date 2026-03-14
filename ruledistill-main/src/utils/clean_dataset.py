#!/usr/bin/env python3
"""
Dataset Cleaner for FinQA train-clear.json

Ensures every sample has valid & consistent `answer` and `exe_ans`.

Strategy:
  1. Fill missing `answer` from `exe_ans` (26 cases)
  2. Multi-strategy matching between `answer` and `exe_ans`:
     - Direct numeric comparison (with tolerance for rounding)
     - Percentage ↔ ratio auto-detection  (e.g. "53%" ≈ 0.532)
     - Percentage-already-multiplied (e.g. "94.9%" ≈ 94.865)
     - ×100 recovery (e.g. "125" ≈ 1.2506 → exe_ans*100 matches answer)
  3. For matched samples: `answer` is the correctness reference,
     `exe_ans` is the more precise ground truth value
  4. Drop samples where answer and exe_ans truly disagree

Usage:
    python clean_dataset.py \\
        --input  /path/to/train-clear.json \\
        --output /path/to/train-clean.json \\
        --report  # print detailed report
"""

import json
import argparse
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------

def _strip_text(raw: str) -> str:
    """Remove currency symbols, units, whitespace, trailing newlines."""
    s = str(raw).strip()
    s = re.sub(r'\\+n$', '', s)
    s = s.replace(",", "").replace("$", "").strip()
    s = re.sub(r'\s*(million|billion|thousand|percent)s?$', '', s, flags=re.I)
    return s.strip()


def _parse_number(s: str):
    """
    Parse a string as a number.

    Returns:
        (float_value, is_percentage)  or  (None, False)
    """
    s = _strip_text(s)
    if not s:
        return None, False

    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1].strip()

    # Parenthesized negative: "(5.2)" -> -5.2
    m = re.match(r'^\((.+)\)$', s)
    if m:
        s = m.group(1)
        try:
            val = -float(s)
        except ValueError:
            return None, is_pct
        return val, is_pct

    # Remove non-numeric chars
    s_clean = re.sub(r'[^\d.\-eE]', '', s)
    try:
        return float(s_clean), is_pct
    except ValueError:
        return None, is_pct


def _close(a: float, b: float, tol: float) -> bool:
    """Check if two floats are within relative tolerance."""
    if a == b:
        return True
    if a == 0 and b == 0:
        return True
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom <= tol


def _match_values(answer_str: str, exe_ans, tolerance: float = 0.10):
    """
    Multi-strategy matching between human `answer` and `exe_ans`.

    Returns:
        (matched: bool, scale: str)
        scale is one of: "direct", "pct_to_ratio", "ratio_to_pct", "times100", None
    """
    if answer_str is None or str(answer_str).strip() == "" or exe_ans is None:
        return False, None

    a_str = str(answer_str).strip()
    e_str = str(exe_ans).strip()

    # ---- Boolean ----
    if a_str.lower() in ("yes", "no") or e_str.lower() in ("yes", "no"):
        return (a_str.lower() == e_str.lower(), "direct")

    # ---- Parse answer ----
    a_val, a_is_pct = _parse_number(a_str)
    if a_val is None:
        # Text answer — try all numbers found in the text against exe_ans
        nums = re.findall(r'[\d]+\.?[\d]*', a_str)
        if not nums:
            return False, None
        try:
            e_val = float(exe_ans)
        except (ValueError, TypeError):
            return False, None
        for n_str in nums:
            try:
                n_val = float(n_str)
            except ValueError:
                continue
            # Try direct match
            if _close(n_val, e_val, tolerance):
                return True, "text_number"
            # Try as percentage (n_val% → n_val/100)
            if _close(n_val / 100.0, e_val, tolerance):
                return True, "text_number"
            # Try ×100
            if _close(n_val * 100.0, e_val, tolerance):
                return True, "text_number"
            if _close(n_val, e_val * 100, tolerance):
                return True, "text_number"
        return False, None

    try:
        e_val = float(exe_ans)
    except (ValueError, TypeError):
        return False, None

    # Strategy 1: Answer has %, convert to ratio, compare with exe_ans
    # e.g. "53%" → 0.53, exe_ans = 0.532
    if a_is_pct:
        a_ratio = a_val / 100.0
        if _close(a_ratio, e_val, tolerance):
            return True, "pct_to_ratio"
        # Also check if exe_ans is already percentage-scaled
        # e.g. "94.9%" → 94.9 vs exe_ans = 94.865
        if _close(a_val, e_val, tolerance):
            return True, "direct"

    # Strategy 2: Direct comparison (same scale)
    if _close(a_val, e_val, tolerance):
        return True, "direct"

    # Strategy 3: Answer is plain number, exe_ans is a ratio
    # e.g. answer = "27.9", exe_ans = 0.27867 → a_val/100 ≈ e_val
    if not a_is_pct and abs(e_val) < 1 and abs(a_val) > 1:
        if _close(a_val / 100.0, e_val, tolerance):
            return True, "ratio_to_pct"

    # Strategy 4: answer is ratio, exe_ans is percentage-scaled
    # e.g. answer = "0.12", exe_ans = 12.0
    if not a_is_pct and abs(a_val) < 1 and abs(e_val) > 1:
        if _close(a_val * 100.0, e_val, tolerance):
            return True, "times100"

    # Strategy 5: ×100 recovery
    # e.g. answer = "125", exe_ans = 1.2506
    if _close(e_val * 100, a_val, tolerance):
        return True, "times100"

    return False, None


def _format_answer_from_exe(exe_ans) -> str:
    """Format exe_ans into a human-readable answer string."""
    if exe_ans is None:
        return ""
    if isinstance(exe_ans, str):
        return exe_ans
    val = float(exe_ans)
    if val == int(val) and abs(val) >= 1:
        return str(int(val))
    if 0 < abs(val) < 1:
        pct = round(val * 100, 5)
        return f"{pct:g}%"
    return f"{val:g}"


# ---------------------------------------------------------------
# Main cleaning logic
# ---------------------------------------------------------------

def clean_dataset(input_path: str, output_path: str, report: bool = False, tolerance: float = 0.10):
    with open(input_path, "r") as f:
        data = json.load(f)

    total = len(data)
    stats = {
        "total": total,
        "matched": 0,
        "filled_from_exe_ans": 0,
        "dropped_mismatch": 0,
        "dropped_missing": 0,
    }
    scale_counts = {}
    cleaned = []
    filled_details = []
    dropped_details = []

    for i, sample in enumerate(data):
        qa = sample.get("qa", {})
        answer = qa.get("answer", "")
        exe_ans = qa.get("exe_ans")
        program = qa.get("program", "")

        has_answer = answer is not None and str(answer).strip() != ""
        has_exe = exe_ans is not None and str(exe_ans).strip() != ""

        if has_answer and has_exe:
            matched, scale = _match_values(answer, exe_ans, tolerance)
            if matched:
                stats["matched"] += 1
                scale_counts[scale] = scale_counts.get(scale, 0) + 1

                # Use exe_ans as the precise ground truth value
                # but keep answer as the correctness reference
                # Store both: answer (human label) + exe_ans (precise value)
                # answer stays as-is (the correctness reference)
                # exe_ans stays as-is (the precise ground truth)
                cleaned.append(sample)
            else:
                stats["dropped_mismatch"] += 1
                dropped_details.append(
                    f"  [{i}] answer={repr(answer):25s} exe_ans={repr(exe_ans):15s} prog={repr(program[:60])}"
                )

        elif not has_answer and has_exe:
            qa["answer"] = _format_answer_from_exe(exe_ans)
            stats["filled_from_exe_ans"] += 1
            filled_details.append(
                f"  [{i}] → answer='{qa['answer']}' (from exe_ans={repr(exe_ans)})"
            )
            cleaned.append(sample)

        else:
            stats["dropped_missing"] += 1
            dropped_details.append(
                f"  [{i}] DROPPED: answer={repr(answer)}, exe_ans={repr(exe_ans)}"
            )

    # Save
    with open(output_path, "w") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    # Report
    print(f"\n{'='*60}")
    print(f"DATASET CLEANING REPORT")
    print(f"{'='*60}")
    print(f"Input:  {input_path} ({total} samples)")
    print(f"Output: {output_path} ({len(cleaned)} samples)")
    print()
    print(f"  ✅ Matched (answer ≈ exe_ans):  {stats['matched']}")
    for scale, count in sorted(scale_counts.items(), key=lambda x: -x[1]):
        print(f"     └ {scale}: {count}")
    print(f"  🔧 Filled answer from exe_ans:  {stats['filled_from_exe_ans']}")
    print(f"  ❌ Dropped (mismatch):           {stats['dropped_mismatch']}")
    print(f"  ❌ Dropped (both missing):       {stats['dropped_missing']}")
    print(f"{'='*60}")
    print(f"  Kept: {len(cleaned)} / {total}  ({len(cleaned)/total*100:.1f}%)")
    print(f"{'='*60}\n")

    if report:
        if filled_details:
            print("FILLED ANSWERS:")
            for line in filled_details:
                print(line)
            print()
        if dropped_details:
            print(f"DROPPED SAMPLES ({len(dropped_details)}):")
            for line in dropped_details:
                print(line)
            print()

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean FinQA dataset")
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--report", action="store_true", help="Print detailed report")
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="Relative tolerance for numeric matching (default: 0.10 = 10%%)")
    args = parser.parse_args()
    clean_dataset(args.input, args.output, args.report, args.tolerance)
