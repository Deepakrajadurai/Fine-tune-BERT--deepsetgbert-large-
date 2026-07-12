"""
strip_whitespace_artifact.py

Fixes the newline label-leak found in training_pair_v5.csv: AI rows
carry embedded '\\n'/'\\n\\n' paragraph breaks from the original
generation output (66.05% of AI rows), while human rows built by
build_human_dataset.py's chunk_by_target_length() never contain a
newline (0.00% of human rows) since it joins sentences with a single
space. This single feature alone lets a classifier hit ~100% F1
without learning anything about actual AI-vs-human style.

This script normalizes ALL text (both classes) by collapsing any
whitespace run -- including embedded newlines, tabs, and repeated
spaces -- into a single space, then strips leading/trailing whitespace.
This is applied uniformly, not just to one class, so no class-specific
whitespace pattern survives as a residual shortcut.

Row count, column set, and row order are all preserved -- this is a
pure text-field transform, not a re-filter or re-sample, so it does not
touch your existing train/val/test group assignments if you've already
run prepare_v5_dataset.py. If you haven't split yet, run this BEFORE
prepare_v5_dataset.py; if you already have splits, either re-run this
against training_pair_v5.csv and re-split, or run it against each split
file individually (see --input accepting multiple files below).

Usage:
    python strip_whitespace_artifact.py \
        --input Data/Final/training_pair_v5.csv \
        --output Data/Final/training_pair_v5_clean.csv \
        --report whitespace_fix_report.md

    # Or fix already-split files in place (creates *_clean.csv siblings):
    python strip_whitespace_artifact.py \
        --input Data/train.csv Data/val.csv Data/external_val.csv Data/test.csv Data/final_holdout.csv \
        --output-suffix _clean \
        --report whitespace_fix_report.md
"""

import argparse
import csv
import re
import sys
from collections import Counter

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse any run of whitespace (space, tab, newline, carriage
    return, etc.) into a single space, then strip leading/trailing
    whitespace. Applied uniformly to both AI and human text."""
    return WHITESPACE_RE.sub(" ", text).strip()


def audit_whitespace(rows):
    """Report newline/tab prevalence by label BEFORE normalization, so
    the fix's effect is documented, not just assumed."""
    counts = Counter()
    totals = Counter()
    for row in rows:
        label = row.get("label", "?")
        text = row.get("text", "")
        totals[label] += 1
        if "\n" in text:
            counts[(label, "newline")] += 1
        if "\t" in text:
            counts[(label, "tab")] += 1
        if "  " in text:  # double space
            counts[(label, "double_space")] += 1
    return counts, totals


def process_file(input_path, output_path, report_lines):
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    before_counts, totals = audit_whitespace(rows)

    report_lines.append(f"## {input_path}\n")
    report_lines.append(f"- Total rows: {len(rows)}\n")
    report_lines.append("### Before normalization\n")
    for label in sorted(totals):
        n = totals[label]
        nl = before_counts.get((label, "newline"), 0)
        tab = before_counts.get((label, "tab"), 0)
        dsp = before_counts.get((label, "double_space"), 0)
        label_name = "AI" if label == "1" else "human" if label == "0" else label
        report_lines.append(
            f"  - label={label} ({label_name}), n={n}: "
            f"newline={nl} ({nl/n:.2%}), tab={tab} ({tab/n:.2%}), "
            f"double_space={dsp} ({dsp/n:.2%})\n"
        )

    for row in rows:
        row["text"] = normalize_text(row["text"])

    after_counts, _ = audit_whitespace(rows)
    report_lines.append("### After normalization (should all be 0)\n")
    for label in sorted(totals):
        nl = after_counts.get((label, "newline"), 0)
        tab = after_counts.get((label, "tab"), 0)
        dsp = after_counts.get((label, "double_space"), 0)
        report_lines.append(f"  - label={label}: newline={nl}, tab={tab}, double_space={dsp}\n")

    # Sanity check: row count and length-parity should still hold roughly
    # (normalization only removes whitespace runs, doesn't cut content)
    lengths_before = {row["id"]: len(row["text"]) for row in rows}  # post-write, for report only

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"{input_path} -> {output_path}: {len(rows)} rows normalized")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True,
                         help="One or more input CSV paths")
    parser.add_argument("--output", default=None,
                         help="Single output path (only valid with one --input file)")
    parser.add_argument("--output-suffix", default=None,
                         help="Suffix to append before .csv for each input file "
                              "(e.g. '_clean' -> train_clean.csv). Use this for "
                              "multiple --input files.")
    parser.add_argument("--report", default="whitespace_fix_report.md")
    args = parser.parse_args()

    if len(args.input) > 1 and args.output:
        print("--output only valid with a single --input file; use --output-suffix "
              "for multiple files.", file=sys.stderr)
        sys.exit(1)
    if len(args.input) == 1 and not args.output and not args.output_suffix:
        print("Provide either --output or --output-suffix.", file=sys.stderr)
        sys.exit(1)

    report_lines = ["# Whitespace Artifact Fix Report\n"]
    total_rows = 0

    for input_path in args.input:
        if args.output:
            output_path = args.output
        else:
            if input_path.endswith(".csv"):
                output_path = input_path[:-4] + args.output_suffix + ".csv"
            else:
                output_path = input_path + args.output_suffix
        total_rows += process_file(input_path, output_path, report_lines)

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nTotal rows normalized across {len(args.input)} file(s): {total_rows}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
