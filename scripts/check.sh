#!/usr/bin/env bash
# Run the full check suite (same as pre-commit, but can be run manually).
# Usage: ./scripts/check.sh [--fix]
#
# --fix: auto-fix formatting issues instead of just checking

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FIX=false
if [[ "$1" == "--fix" ]]; then
    FIX=true
fi

echo -e "${YELLOW}=== The Dark Candle — Check Suite ===${NC}"
echo ""

# 1. Format
if $FIX; then
    echo -e "${YELLOW}[1/3]${NC} cargo fmt (fixing)"
    cargo fmt
    echo -e "${GREEN}  ✓ Formatted${NC}"
else
    echo -e "${YELLOW}[1/3]${NC} cargo fmt --check"
    if ! cargo fmt --check 2>&1; then
        echo -e "${RED}✗ Formatting issues. Run './scripts/check.sh --fix' to auto-fix.${NC}"
        exit 1
    fi
    echo -e "${GREEN}  ✓ Formatting OK${NC}"
fi

# 2. Clippy
echo -e "${YELLOW}[2/3]${NC} cargo clippy --all-targets"
if ! cargo clippy --all-targets -- -D warnings 2>&1; then
    echo -e "${RED}✗ Clippy warnings found.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Clippy OK${NC}"

# 3. Tests
echo -e "${YELLOW}[3/3]${NC} cargo test"
if ! cargo test 2>&1; then
    echo -e "${RED}✗ Tests failed.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Tests OK${NC}"

echo ""
echo -e "${GREEN}=== All checks passed ===${NC}"
