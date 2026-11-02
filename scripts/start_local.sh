#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p workspace
BIN=.runtime/llama-b11744/llama-server
if [ ! -x "$BIN" ]; then echo '먼저 sh scripts/download_local.sh를 실행하세요.'; exit 1; fi
"$BIN" -m .runtime/models/qwen2.5-0.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8765 --alias ysc-qwen-0.5b --ctx-size 8192 --parallel 1 --threads 4 > workspace/model.log 2>&1 &
YSC_GEN_PID=$!
"$BIN" -m .runtime/models/nomic-embed-text-v1.5.Q4_K_M.gguf --embedding --pooling mean --host 127.0.0.1 --port 8766 --alias ysc-nomic-embed --ctx-size 2048 --parallel 1 --threads 4 > workspace/embed.log 2>&1 &
YSC_EMB_PID=$!
trap 'kill "$YSC_GEN_PID" "$YSC_EMB_PID" 2>/dev/null || true' EXIT INT TERM
echo '로컬 모델을 시작했습니다. 로그: workspace/model.log, workspace/embed.log'
echo '이 터미널을 열어 두고, 다른 터미널에서 앱/배치를 실행하세요. 종료: Ctrl+C'
wait "$YSC_GEN_PID" "$YSC_EMB_PID"
