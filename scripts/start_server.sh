#!/usr/bin/env bash
# =====================================================================
#  start_server.sh  --  Khởi chạy llama-cpp-turboquant server (v2)
# =====================================================================
#  Mục đích: linh hoạt khởi chạy server ở 2 chế độ, đều load CÙNG một file
#  GGUF nằm trong Factaudit/models/ (ví dụ: Qwen3-14B-Q8_0.gguf).
#
#  Usage:
#     ./scripts/start_server.sh              # = both (chạy cả 2 server nền)
#     ./scripts/start_server.sh baseline     # port 8080, cache f32
#     ./scripts/start_server.sh turboquant   # port 8081, cache turbo3/turbo4
#     ./scripts/start_server.sh both         # chạy cả 2 (background + log)
#
#  Override cấu hình qua biến môi trường (có giá trị mặc định):
#     LLAMA_CPP_DIR        thư mục repo llama-cpp-turboquant
#     LLAMA_SERVER_BIN     đường dẫn binary server (mặc định <DIR>/build/bin/llama-server)
#     GGUF_MODEL           đường dẫn file GGUF (mặc định <project>/models/Qwen3-14B-Q8_0.gguf)
#     MODEL_ALIAS          alias gửi qua API (mặc định = tên file GGUF bỏ đuôi)
#     GPU_LAYERS / THREADS / HOST / *_PORT / *_CTX ...
#
#  Lưu ý: bản build cũ của llama.cpp đặt tên binary là `server` (đường dẫn
#         build/bin/server). Bản mới dùng `llama-server`. Nếu báo "command
#         not found", hãy set:  export LLAMA_SERVER_BIN=/path/to/server
# =====================================================================
set -euo pipefail

# ---------------------------------------------------------------------
# Resolve đường dẫn (script nằm trong Factaudit/scripts/)
# ---------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"

# Repo llama-cpp-turboquant (mặc định là anh em với Factaudit)
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PROJECT_ROOT/../llama-cpp-turboquant}"
# Binary server: bản mới là llama-server, bản cũ là server
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_CPP_DIR/build/bin/llama-server}"

# File GGUF
GGUF_MODEL="${GGUF_MODEL:-$MODELS_DIR/Qwen3-14B-Q8_0.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-$(basename "$GGUF_MODEL" .gguf)}"

# Tham số server chung
THREADS="${THREADS:-8}"
GPU_LAYERS="${GPU_LAYERS:-35}"
HOST="${HOST:-0.0.0.0}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"

# Port & context cho từng mode
BASELINE_PORT="${BASELINE_PORT:-8080}"
TURBOQUANT_PORT="${TURBOQUANT_PORT:-8081}"
BASELINE_CTX="${BASELINE_CTX:-8192}"
TURBOQUANT_CTX="${TURBOQUANT_CTX:-32768}"

# TurboQuant specifics
TURBO_K="${TURBO_K:-turbo3}"
TURBO_V="${TURBO_V:-turbo4}"
TURBO_QUANT_BITS="${TURBO_QUANT_BITS:-4}"
TURBO_GROUP_SIZE="${TURBO_GROUP_SIZE:-64}"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
c_red()   { printf '\033[31m%s\033[0m' "$*"; }
c_green() { printf '\033[32m%s\033[0m' "$*"; }
c_cyan()  { printf '\033[36m%s\033[0m' "$*"; }

banner() {
  echo
  echo "$(c_cyan '==================================================')"
  echo "  $1"
  echo "$(c_cyan '==================================================')"
}

preflight() {
  local missing=0
  if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
    # Cho phép file tồn tại nhưng không có +x trên Windows/MSYS
    if [[ ! -f "$LLAMA_SERVER_BIN" ]]; then
      echo "$(c_red '[ERROR]') Không tìm thấy server binary:"
      echo "    $LLAMA_SERVER_BIN"
      echo "  -> Build llama-cpp-turboquant trước, hoặc set LLAMA_SERVER_BIN."
      echo "     (bản build cũ dùng tên 'server' thay cho 'llama-server')"
      missing=1
    fi
  fi
  if [[ ! -f "$GGUF_MODEL" ]]; then
    echo "$(c_red '[ERROR]') Không tìm thấy file GGUF:"
    echo "    $GGUF_MODEL"
    echo "  -> Tải model về Factaudit/models/ hoặc set GGUF_MODEL=<đường dẫn>."
    missing=1
  fi
  if [[ "$missing" -ne 0 ]]; then
    exit 1
  fi
}

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 1
}

# ---------------------------------------------------------------------
# Khởi chạy 1 server. Background nếu $2 == "bg".
#   $1 = tag (baseline|turboquant), $2 = mode chạy (bg|fg)
# ---------------------------------------------------------------------
run_server() {
  local tag="$1"
  local bg="${2:-fg}"
  shift 2 || true
  local log_file="$LOG_DIR/server_${tag}.log"
  local pid_file="$LOG_DIR/server_${tag}.pid"

  if [[ "$bg" == "bg" ]]; then
    echo "$(c_green "[start]") $tag -> background"
    echo "    cmd : $*"
    echo "    log : $log_file"
    "$@" > "$log_file" 2>&1 &
    echo $! > "$pid_file"
    echo "    pid : $(cat "$pid_file")"
  else
    banner "Server: $tag  (Ctrl+C để dừng)"
    echo "  cmd: $*"
    echo "  log: $log_file  (tee song song terminal)"
    echo
    "$@" 2>&1 | tee "$log_file"
  fi
}

start_baseline() {
  run_server "baseline" "$1" \
    "$LLAMA_SERVER_BIN" \
      --host "$HOST" \
      --port "$BASELINE_PORT" \
      --model "$GGUF_MODEL" \
      --alias "$MODEL_ALIAS" \
      --threads "$THREADS" \
      --n-gpu-layers "$GPU_LAYERS" \
      --ctx-size "$BASELINE_CTX" \
      --batch-size 512 \
      --ubatch-size 128 \
      --cache-type-k f32 \
      --cache-type-v f32 \
      --metrics \
      --log-format text
}

start_turboquant() {
  run_server "turboquant" "$1" \
    "$LLAMA_SERVER_BIN" \
      --host "$HOST" \
      --port "$TURBOQUANT_PORT" \
      --model "$GGUF_MODEL" \
      --alias "$MODEL_ALIAS" \
      --threads "$THREADS" \
      --n-gpu-layers "$GPU_LAYERS" \
      --ctx-size "$TURBOQUANT_CTX" \
      --batch-size 1024 \
      --ubatch-size 256 \
      --cache-type-k "$TURBO_K" \
      --cache-type-v "$TURBO_V" \
      --turbo-quant-bits "$TURBO_QUANT_BITS" \
      --turbo-group-size "$TURBO_GROUP_SIZE" \
      --metrics \
      --log-format text
}

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
preflight
ACTION="${1:-both}"

case "$ACTION" in
  baseline)
    start_baseline fg
    ;;
  turboquant|turbo)
    start_turboquant fg
    ;;
  both)
    banner "Khởi chạy CẢ 2 server ở nền (background)"
    start_baseline bg
    start_turboquant bg
    echo
    echo "$(c_green '[ok]') Đã khởi động cả 2 server."
    echo "  Baseline    : http://localhost:$BASELINE_PORT/v1/models"
    echo "  TurboQuant+ : http://localhost:$TURBOQUANT_PORT/v1/models"
    echo
    echo "  Xem log     : tail -f $LOG_DIR/server_baseline.log"
    echo "                tail -f $LOG_DIR/server_turboquant.log"
    echo
    echo "  Dừng        : kill \$(cat $LOG_DIR/server_*.pid)"
    echo
    echo "  Health check:"
    echo "    curl -s http://localhost:$BASELINE_PORT/health  && echo '  baseline ok'"
    echo "    curl -s http://localhost:$TURBOQUANT_PORT/health && echo '  turboquant ok'"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "$(c_red "[ERROR]") Action không hợp lệ: $ACTION"
    usage
    ;;
esac
