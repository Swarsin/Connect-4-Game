#!/usr/bin/env bash
# Build Pascal Pons' Connect 4 solver (AGPL-3.0, used only as an external evaluation tool;
# never committed to this repo or shipped with the website) and download its opening book.
#
# Usage: scripts/build_solver.sh [--small-book]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/connect4-solver"
BOOK_URL="https://github.com/PascalPons/connect4/releases/download/book/7x6.book"
if [[ "${1:-}" == "--small-book" ]]; then
  BOOK_URL="https://github.com/PascalPons/connect4/releases/download/book/7x6_small.book"
fi

mkdir -p "$ROOT/third_party"
if [[ ! -d "$DEST/.git" ]]; then
  git clone --depth 1 https://github.com/PascalPons/connect4.git "$DEST"
fi

make -C "$DEST" c4solver CXXFLAGS="--std=c++11 -O3 -DNDEBUG -march=native"

if [[ ! -f "$DEST/7x6.book" ]]; then
  echo "Downloading opening book from $BOOK_URL"
  curl -L --fail -o "$DEST/7x6.book" "$BOOK_URL"
fi

echo "Solver ready: $DEST/c4solver (book: $DEST/7x6.book)"
echo "4" | (cd "$DEST" && ./c4solver -a)
