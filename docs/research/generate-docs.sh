#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Generating documentation ==="

echo "Generating algorithms.docx..."
node algorithms.js

echo "Generating paper.docx..."
node paper.js

echo "Converting to markdown..."
pandoc algorithms.docx -o algorithms.md --wrap=none
pandoc paper.docx -o paper.md --wrap=none

echo ""
echo "=== Checking for duplicate section anchors ==="

check_dups() {
    local file="$1"
    local label="$2"

    # Duplicate section numbers (e.g., two different "3.1.1 ..." headings)
    local dup_nums
    dup_nums=$(grep -E '^[0-9]+\.[0-9.]* [A-Z]' "$file" | awk '{print $1}' | sort | uniq -d || true)
    if [[ -n "$dup_nums" ]]; then
        echo "ERROR: Duplicate section numbers found in $label:"
        echo "$dup_nums" | sed 's/^/  - /'
        exit 1
    fi

    # Duplicate exact heading lines
    local dup_lines
    dup_lines=$(grep -E '^[0-9]+\.[0-9.]* [A-Z]' "$file" | sort | uniq -d || true)
    if [[ -n "$dup_lines" ]]; then
        echo "ERROR: Duplicate section headings found in $label:"
        echo "$dup_lines" | sed 's/^/  - /'
        exit 1
    fi
}

check_dups "algorithms.md" "algorithms.md"
check_dups "paper.md" "paper.md"

echo ""
echo "=== Checking for forbidden / deprecated tokens in generated markdown ==="

check_forbidden() {
    local file="$1"
    local label="$2"
    shift 2

    local bad=0
    for token in "$@"; do
        if grep -nF -- "$token" "$file" >/dev/null; then
            echo "ERROR: Found forbidden token '$token' in $label"
            grep -nF -- "$token" "$file" | head -n 5 | sed 's/^/  /'
            bad=1
        fi
    done

    if [[ "$bad" -ne 0 ]]; then
        exit 1
    fi
}

# These must not appear anywhere in the generated paper/spec.
check_forbidden "paper.md" "paper.md" \
  "hysteresis_band" \
  "last_rate_ts" \
  "t_acc_start" \
  "D_acc" \
  "η_mem" \
  "last_signal_timestamp" \
  "co_occurs_with" \
  "implies/causes"

check_forbidden "algorithms.md" "algorithms.md" \
  "hysteresis_band" \
  "last_rate_ts" \
  "t_acc_start" \
  "D_acc" \
  "η_mem" \
  "last_signal_timestamp" \
  "co_occurs_with" \
  "implies/causes"

# Ensure we have exactly one canonical contract + one units section in each generated markdown.
assert_count_eq() {
    local file="$1"
    local label="$2"
    local needle="$3"
    local expected="$4"
    local count
    count=$(grep -cF -- "$needle" "$file" || true)
    if [[ "$count" -ne "$expected" ]]; then
        echo "ERROR: Expected $expected occurrence(s) of '$needle' in $label, found $count"
        grep -nF -- "$needle" "$file" | head -n 10 | sed 's/^/  /' || true
        exit 1
    fi
}

assert_count_eq "paper.md" "paper.md" "Naming contract (canonical):" 1
assert_count_eq "paper.md" "paper.md" "Units Convention" 1
assert_count_eq "paper.md" "paper.md" "Memory Accumulator State" 1
assert_count_eq "paper.md" "paper.md" "Write Exclusion Filter" 1
assert_count_eq "algorithms.md" "algorithms.md" "Naming contract (canonical):" 1
assert_count_eq "algorithms.md" "algorithms.md" "Units Convention" 1
assert_count_eq "algorithms.md" "algorithms.md" "Memory Accumulator State" 1
assert_count_eq "algorithms.md" "algorithms.md" "Write Exclusion Filter" 1

# Special-case: forbid now() specifically (now_s()/now_ms() are allowed)
if grep -nE -- 'now\(\)' paper.md algorithms.md >/dev/null; then
    echo "ERROR: Found forbidden call-like token now() in generated markdown"
    grep -nE -- 'now\(\)' paper.md algorithms.md | head -n 10 | sed 's/^/  /'
    exit 1
fi

echo ""
echo "=== Validating paper includes all algorithm sections ==="

MISSING=0
SECTION_OFFSET=2

while IFS= read -r line; do
    section_num=$(echo "$line" | grep -oE '^[0-9]+\.[0-9.]*' | head -1)
    section_title=$(echo "$line" | sed -E 's/^[0-9]+\.[0-9.]* //')
    
    if [[ -z "$section_num" ]]; then
        continue
    fi
    
    major=$(echo "$section_num" | cut -d. -f1)
    rest=$(echo "$section_num" | cut -d. -f2-)
    paper_major=$((major + SECTION_OFFSET))
    
    if [[ -n "$rest" ]]; then
        paper_section="${paper_major}.${rest}"
    else
        paper_section="${paper_major}"
    fi
    
    # Must exist exactly once; both missing and duplicated anchors are failures.
    count=$(grep -cF "$paper_section $section_title" paper.md || true)
    if [[ "$count" -eq 0 ]]; then
        echo "MISSING: Section $section_num '$section_title' (expected $paper_section in paper)"
        MISSING=$((MISSING + 1))
    elif [[ "$count" -gt 1 ]]; then
        echo "DUPLICATE: Section $section_num '$section_title' appears $count times in paper (expected exactly 1)"
        MISSING=$((MISSING + 1))
    fi
done < <(grep -E '^[0-9]+\.[0-9.]* [A-Z]' algorithms.md)

echo ""
if [[ $MISSING -eq 0 ]]; then
    echo "OK: All algorithm sections found in paper (with +$SECTION_OFFSET offset)"
else
    echo "ERROR: $MISSING section(s) missing from paper"
    exit 1
fi

echo ""
echo "=== Generated files ==="
ls -lh algorithms.docx algorithms.md paper.docx paper.md
