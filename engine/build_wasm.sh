#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p frontend/public/engine
em++ -std=c++20 -O3 -Wall -Wextra -Werror -fexceptions --no-entry \
  -sDISABLE_EXCEPTION_CATCHING=0 -sMODULARIZE=1 -sEXPORT_ES6=1 \
  -sENVIRONMENT=web,worker -sALLOW_MEMORY_GROWTH=1 -sFORCE_FILESYSTEM=1 \
  -sEXPORTED_FUNCTIONS='["_dc_init","_dc_analyze"]' \
  -sEXPORTED_RUNTIME_METHODS='["FS","ccall"]' \
  engine/wasm_bridge.cpp -o frontend/public/engine/darkchess.mjs
