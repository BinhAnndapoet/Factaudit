# Hướng dẫn chạy FACT-AUDIT

Tài liệu này hướng dẫn cách cài đặt và chạy hệ thống **FACT-AUDIT** theo **2 chế độ (mode)**:

- **Baseline** — chạy KV cache ở chế độ `f32` (độ chính xác đầy đủ), context 8K tokens.
- **TurboQuant+** — chạy KV cache nén (`turbo3`/`turbo4`), context 32K tokens (gấp 4×), tiết kiệm VRAM.

> Mode-switching là **toàn cục**: một tham số `--mode` sẽ flip đồng thời cả **Model A** (5 agent) và **Model B** (target model). Không có mode riêng cho từng model.

---

## Mục lục

1. [Kiến trúc tóm tắt](#1-kiến-trúc-tóm-tắt)
2. [Yêu cầu hệ thống](#2-yêu-cầu-hệ-thống)
3. [Cài đặt](#3-cài-đặt)
4. [Cấu hình `.env`](#4-cấu-hình-env)
5. [Kiểm tra trước khi chạy](#5-kiểm-tra-trước-khi-chạy)
6. [Chạy theo Mode Baseline](#6-chạy-theo-mode-baseline)
7. [Chạy theo Mode TurboQuant+](#7-chạy-theo-mode-turboquant)
8. [So sánh hai mode](#8-so-sánh-hai-mode)
9. [Đọc kết quả & metrics](#9-đọc-kết-quả--metrics)
10. [Các lệnh phụ trợ](#10-các-lệnh-phụ-trợ)
11. [Khắc phục sự cố thường gặp](#11-khắc-phục-sự-cố-thường-gặp)

---

## 1. Kiến trúc tóm tắt

Hệ thống gồm **2 model độc lập**, mỗi model được serve bởi **một cặp server** llama-cpp-turboquant (1 baseline + 1 turboquant) → tổng cộng **4 server**:

| Vai trò | Model GGUF (thiết kế) | Baseline (f32) | TurboQuant+ (turbo) | Dùng cho |
|---------|----------------------|----------------|---------------------|----------|
| **Model A** | `Qwen3-32B-Q8_0.gguf` | port **8080** | port **8081** | 5 agent (Appraiser, Inquirer, Quality Inspector, Evaluator, Prober) |
| **Model B** | `Qwen3-14B-Q8_0.gguf` | port **8082** | port **8083** | Target Model (mô hình bị kiểm toán) |

- Khi chạy `--mode baseline`: dùng **A:8080 + B:8082**.
- Khi chạy `--mode turboquant`: dùng **A:8081 + B:8083**.

> ⚠️ **Lưu ý về model hiện có:** Theo thiết kế, Model A = `Qwen3-32B`. Tuy nhiên thư mục `models/` hiện **chỉ có `Qwen3-14B-Q8_0.gguf`** (file 32B chưa được tải về). Vì vậy file `.env` hiện đang cấu hình **Model A cũng dùng `Qwen3-14B-Q8_0.gguf`** để vẫn chạy được. Nếu muốn dùng đúng thiết kế (Model A = 32B), hãy tải thêm `Qwen3-32B-Q8_0.gguf` vào `models/` rồi mở comment các dòng `MODEL_A_*` tương ứng trong `.env`.

---

## 2. Yêu cầu hệ thống

### Phần cứng (GPU)

Khi offload đầy đủ lên GPU (`GPU_LAYERS=35`), script ước lượng VRAM:

| Cấu hình | Ước lượng VRAM |
|----------|----------------|
| 2 server baseline (A+B) | ≈ 30 GB (với 14B cho cả 2 model) |
| 4 server cùng lúc (`both`) | ≈ 60–96 GB+ → chỉ phù hợp rig lớn |

Nếu thiếu VRAM, hạ `GPU_LAYERS` (xem [§11](#11-khắc-phục-sự-cố-thường-gặp)) hoặc chạy từng server (action granular như `a-turbo`).

### Phần mềm

- **OS**: Linux (đã test trên kernel 6.17).
- **Python**: 3.10 (qua conda env `turboquant`).
- **llama-cpp-turboquant**: repo anh em `../llama-cpp-turboquant`, đã build binary `llama-server`.

---

## 3. Cài đặt

### 3.1. Build binary `llama-server` (chỉ làm 1 lần)

```bash
cd ../llama-cpp-turboquant
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON        # bật CUDA nếu có GPU NVIDIA
cmake --build . --config Release -j
```

Kiểm tra binary đã sinh ra:

```bash
ls -la ../llama-cpp-turboquant/build/bin/llama-server
```

> Nếu bản build cũ đặt tên binary là `server` (đường dẫn `build/bin/server`), hãy set:
> ```bash
> export LLAMA_SERVER_BIN=/path/to/server
> ```

### 3.2. Tạo môi trường Python

Yêu cầu conda env tên **`turboquant`** (đã được dùng trong các log chạy thật):

```bash
conda create -n turboquant python=3.10 -y
conda activate turboquant
```

### 3.3. Cài Python dependencies

```bash
cd /home/guest/Projects/ban/Factaudit
pip install -r src/requirements.txt
```

Các thư viện chính: `langgraph`, `langchain-openai`, `langchain-google-genai`, `langchain-tavily`, `pydantic`, `python-dotenv`, `pandas`, `langsmith`.

### 3.4. Tải model GGUF

Đặt các file `.gguf` vào thư mục `models/`:

```bash
Factaudit/
└── models/
    ├── Qwen3-14B-Q8_0.gguf   ← bắt buộc (đã có sẵn, 15 GB)
    └── Qwen3-32B-Q8_0.gguf   ← tùy chọn (Model A theo thiết kế)
```

---

## 4. Cấu hình `.env`

File `.env` nằm ở thư mục gốc `Factaudit/`. Các mục quan trọng:

```bash
# --- API keys ---
TAVILY_API_KEY="tvly-..."        # BẮT BUỘC (xác minh evidence qua web/Wikipedia)
GEMINI_API_KEY=                  # Có thể bỏ trống (fallback cloud, không bắt buộc)

# --- LLM MODE ---
MODE=auto                        # auto | baseline | turboquant
USE_TURBOQUANT=false             # Chỉ tác động khi MODE=auto (true => turboquant)

# --- Model A (5 agent) ---
MODEL_A_ALIAS=Qwen3-14B-Q8_0
MODEL_A_GGUF_FILE=Qwen3-14B-Q8_0.gguf
MODEL_A_BASELINE_API_BASE=http://127.0.0.1:8080/v1
MODEL_A_TURBOQUANT_API_BASE=http://127.0.0.1:8081/v1

# --- Model B (target) ---
MODEL_B_ALIAS=Qwen3-14B-Q8_0
MODEL_B_GGUF_FILE=Qwen3-14B-Q8_0.gguf
MODEL_B_BASELINE_API_BASE=http://127.0.0.1:8082/v1
MODEL_B_TURBOQUANT_API_BASE=http://127.0.0.1:8083/v1

# --- Context size ---
MAX_CONTEXT_SIZE=8192            # baseline
TURBOQUANT_CONTEXT_SIZE=32768    # turboquant

# --- Tracing LangSmith (tuỳ chọn) ---
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="lsv2_pt_..."
LANGCHAIN_PROJECT="FactAudit_Agent_Simulation"
```

**Thứ tự ưu tiên chọn mode** (cao → thấp):

1. Tham số `--mode baseline|turboquant` trên dòng lệnh.
2. Biến `MODE` trong `.env`.
3. Cờ `USE_TURBOQUANT` (chỉ khi `MODE=auto`).

> Khi truyền `--mode baseline` hoặc `--mode turboquant` thì các giá trị trong `.env` sẽ bị ghi đè.

---

## 5. Kiểm tra trước khi chạy

**Luôn chạy từ thư mục gốc `Factaudit/`** (vì `main.py` dùng `load_dotenv()` và các đường dẫn tương đối):

```bash
cd /home/guest/Projects/ban/Factaudit
conda activate turboquant
```

Kiểm tra nhanh:

```bash
# Binary server tồn tại?
ls -la ../llama-cpp-turboquant/build/bin/llama-server

# Model GGUF tồn tại?
ls -lh models/*.gguf
```

---

## 6. Chạy theo Mode Baseline

### Bước 1 — Khởi động 2 server baseline (A:8080 + B:8082)

**Cách 1 — Dùng script (khuyến nghị, tự log + pid):**

```bash
cd /home/guest/Projects/ban/Factaudit
./scripts/start_server.sh baseline
```

Script chạy server ở **background** và ghi:

- log: `logs/server_a_baseline.log`, `logs/server_b_baseline.log`
- pid: `logs/server_a_baseline.pid`, `logs/server_b_baseline.pid`

**Cách 2 — Chạy trực tiếp `./bin/llama-server` (không qua script):**

Di chuyển vào thư mục `build` của `llama-cpp-turboquant` rồi gọi binary, mỗi model một server (cửa sổ riêng hoặc thêm `&` để chạy nền):

```bash
cd /home/guest/Projects/ban/llama-cpp-turboquant/build
MODEL=/home/guest/Projects/ban/Factaudit/models/Qwen3-14B-Q8_0.gguf
LOGS=/home/guest/Projects/ban/Factaudit/logs

# Model A (5 agent) — baseline, port 8080
./bin/llama-server \
  --host 0.0.0.0 --port 8080 \
  --model "$MODEL" --alias Qwen3-14B-Q8_0 \
  --threads 8 --n-gpu-layers 35 \
  --ctx-size 8192 --batch-size 512 --ubatch-size 128 \
  --cache-type-k f32 --cache-type-v f32 \
  --metrics --log-format text > "$LOGS/server_a_baseline.log" 2>&1 &

# Model B (target) — baseline, port 8082
./bin/llama-server \
  --host 0.0.0.0 --port 8082 \
  --model "$MODEL" --alias Qwen3-14B-Q8_0 \
  --threads 8 --n-gpu-layers 35 \
  --ctx-size 8192 --batch-size 512 --ubatch-size 128 \
  --cache-type-k f32 --cache-type-v f32 \
  --metrics --log-format text > "$LOGS/server_b_baseline.log" 2>&1 &
```

> Các tham số này khớp 100% với `start_server.sh baseline`. Bỏ `&` và `> ...` nếu muốn xem log trực tiếp trên terminal. Để chuyển sang TurboQuant, đổi các cờ theo [§7 — Cách 2](#7-chạy-theo-mode-turboquant).
>
> Nếu có model 32B cho Model A, thay `--model` của server 8080 bằng `.../models/Qwen3-32B-Q8_0.gguf` và `--alias Qwen3-32B-Q8_0`.

Chờ tới khi cả hai server báo sẵn sàng (kiểm tra health):

```bash
curl -s http://127.0.0.1:8080/health && echo "  A baseline ok"
curl -s http://127.0.0.1:8082/health && echo "  B baseline ok"
```

> Khi thấy dòng `server: listening` / `all slots are ready` trong log là server đã sẵn sàng.

### Bước 2 — Chạy FACT-AUDIT ở mode baseline

```bash
python src/main.py --mode baseline
```

Banner sẽ hiển thị `Mode: BASELINE (Q8_0/FP16)`, `Max Context: 8192 tokens`, và endpoint của Model A/B trỏ tới `:8080` / `:802`.

Lúc này log chạy được ghi song song ra terminal **và** file `logs/fact_audit_BASE_<timestamp>.log`.

---

## 7. Chạy theo Mode TurboQuant+

### Bước 1 — Khởi động 2 server turboquant (A:8081 + B:8083)

**Cách 1 — Dùng script (khuyến nghị, tự log + pid):**

```bash
cd /home/guest/Projects/ban/Factaudit
./scripts/start_server.sh turboquant
```

Các cờ TurboQuant được bật thêm: `--turbo-quant-bits 4`, `--turbo-group-size 64`, cache `K=turbo3 V=turbo4`, context 32K, batch lớn hơn (1024/256).

**Cách 2 — Chạy trực tiếp `./bin/llama-server` (không qua script):**

Khác baseline ở 4 điểm: `--ctx-size 32768`, `--batch-size 1024 --ubatch-size 256`, `--cache-type-k turbo3 --cache-type-v turbo4`, và thêm `--turbo-quant-bits 4 --turbo-group-size 64`:

```bash
cd /home/guest/Projects/ban/llama-cpp-turboquant/build
MODEL=/home/guest/Projects/ban/Factaudit/models/Qwen3-14B-Q8_0.gguf
LOGS=/home/guest/Projects/ban/Factaudit/logs

# Model A (5 agent) — turboquant, port 8081
./bin/llama-server \
  --host 0.0.0.0 --port 8081 \
  --model "$MODEL" --alias Qwen3-14B-Q8_0 \
  --threads 8 --n-gpu-layers 35 \
  --ctx-size 32768 --batch-size 1024 --ubatch-size 256 \
  --cache-type-k turbo3 --cache-type-v turbo4 \
  --turbo-quant-bits 4 --turbo-group-size 64 \
  --metrics --log-format text > "$LOGS/server_a_turbo.log" 2>&1 &

# Model B (target) — turboquant, port 8083
./bin/llama-server \
  --host 0.0.0.0 --port 8083 \
  --model "$MODEL" --alias Qwen3-14B-Q8_0 \
  --threads 8 --n-gpu-layers 35 \
  --ctx-size 32768 --batch-size 1024 --ubatch-size 256 \
  --cache-type-k turbo3 --cache-type-v turbo4 \
  --turbo-quant-bits 4 --turbo-group-size 64 \
  --metrics --log-format text > "$LOGS/server_b_turbo.log" 2>&1 &
```

> Các cờ `--cache-type-k/v` và `--turbo-quant-*` là của bản fork `llama-cpp-turboquant` (bản `llama.cpp` chính gốc không có) → bắt buộc phải build đúng repo anh em.

Health check:

```bash
curl -s http://127.0.0.1:8081/health && echo "  A turbo ok"
curl -s http://127.0.0.1:8083/health && echo "  B turbo ok"
```

### Bước 2 — Chạy FACT-AUDIT ở mode turboquant

```bash
python src/main.py --mode turboquant
```

Banner hiển thị `Mode: TURBOQUANT+ (KV Cache Compression)`, `Max Context: 32768 tokens (4x capacity)`, endpoint trỏ tới `:8081` / `:8083`.

Log chạy: `logs/fact_audit_TURB_<timestamp>.log`.

---

### Tóm tắt lệnh (copy-paste)

```bash
# === BASELINE ===
cd /home/guest/Projects/ban/Factaudit && conda activate turboquant
./scripts/start_server.sh baseline      # server A:8080 + B:8082 (background)
# ...đợi server ready, kiểm tra bằng curl /health...
python src/main.py --mode baseline

# === TURBOQUANT ===
cd /home/guest/Projects/ban/Factaudit && conda activate turboquant
./scripts/start_server.sh turboquant    # server A:8081 + B:8083 (background)
# ...đợi server ready...
python src/main.py --mode turboquant
```

### Một số biến thể hữu ích của `main.py`

```bash
python src/main.py                            # AUTO: theo MODE/USE_TURBOQUANT trong .env
python src/main.py --mode turboquant -i 5     # tăng số vòng Prober lên 5 (mặc định 3, paper dùng 30)
python src/main.py --mode baseline -v         # verbose: in chi tiết từng node
python src/main.py --no-log                   # chỉ in ra terminal, không ghi file log
```

---

## 8. So sánh hai mode

| Tiêu chí | Baseline | TurboQuant+ |
|----------|----------|-------------|
| KV cache | `f32` (đầy đủ) | `turbo3`/`turbo4` (nén) |
| Max context | 8 192 tokens | 32 768 tokens (×4) |
| Port Model A / B | 8080 / 8082 | 8081 / 8083 |
| Batch / ubatch | 512 / 128 | 1024 / 256 |
| Cờ TurboQuant | (không) | `--turbo-quant-bits 4 --turbo-group-size 64` |
| Cờ mode khi chạy | `--mode baseline` | `--mode turboquant` |
| Tiêu chí chọn | Đánh giá nền tảng, độ chính xác cao nhất | Bài toán context dài, tiết kiệm VRAM, benchmark nén |

---

## 9. Đọc kết quả & metrics

Khi chạy xong, `main.py` tự gọi `compute_metrics.py` đọc `memory_pool.json` và in 3 chỉ số:

- **Grade** — điểm trung bình (trên 10) của target model.
- **IMR** (Insight Mastery Rate) — % test case bị đánh giá yếu (score ≤ 3.0).
- **JFR** (Justification Flaw Rate) — % case target ra **đúng kết luận** nhưng **lý giải sai/yếu**.

Các đầu ra:

```bash
logs/fact_audit_{BASE|TURB}_<timestamp>.log   # toàn bộ log chạy
memory_pool.json                               # dữ liệu thô từng case (nếu có)
img/fact_audit_architecture.png                # sơ đồ kiến trúc (sinh bởi visualize_graph.py)
```

Chạy lại metrics thủ công (không cần chạy lại cả graph):

```bash
python compute_metrics.py
```

---

## 10. Các lệnh phụ trợ

### 10.1. Khởi động server ở các chế độ khác

```bash
./scripts/start_server.sh            # = both: cả 4 server (cần VRAM lớn)
./scripts/start_server.sh model-a     # cả 2 server Model A (8080 + 8081)
./scripts/start_server.sh model-b     # cả 2 server Model B (8082 + 8083)
./scripts/start_server.sh a-baseline  # chạy 1 server ở foreground (dev, VRAM thấp)
./scripts/start_server.sh a-turbo
./scripts/start_server.sh b-baseline
./scripts/start_server.sh b-turbo
```

### 10.2. Theo dõi log server

```bash
tail -f logs/server_a_baseline.log logs/server_b_baseline.log
```

### 10.3. Dừng toàn bộ server

```bash
# Khởi động qua script (có file .pid):
kill $(cat logs/server_*.pid)

# Khởi động thủ công bằng ./bin/llama-server (không có .pid):
pkill -f llama-server
# hoặc dừng theo port (baseline 8080/8082, turboquant 8081/8083):
kill $(lsof -t -i:8080 -i:8082)
```

### 10.4. Vẽ sơ đồ kiến trúc (PNG)

```bash
python src/visualize_graph.py        # xuất img/fact_audit_architecture.png
```

---

## 11. Khắc phục sự cố thường gặp

| Triệu chứng | Nguyên nhân & cách xử lý |
|-------------|--------------------------|
| `[ERROR] Không tìm thấy server binary` | Chưa build `llama-server`, hoặc set `export LLAMA_SERVER_BIN=/path/to/llama-server`. |
| `[ERROR] Không tìm thấy file GGUF` | Thiếu file `.gguf` trong `models/`. Tải về hoặc set `MODEL_A_GGUF_FILE`/`MODEL_B_GGUF_FILE` đúng tên file. |
| `Connection error` / `Max retries` khi gọi LLM | Server chưa sẵn sàng, sai port, hoặc mode không khớp. Kiểm tra `--mode` phải đúng với server đã start (baseline→8080/8082, turboquant→8081/8083). |
| `[WebTool] Error during web search` | `TAVILY_API_KEY` sai/hết quota, hoặc không có mạng. Các node Inspector vẫn tiếp tục với dữ liệu bảo lưu. |
| Server crash / OOM khi load model | Hạ số layer offload: `GPU_LAYERS=20 ./scripts/start_server.sh baseline`, hoặc giảm `BASELINE_CTX`/`TURBOQUANT_CTX`, hoặc chạy từng server (`a-turbo`...). |
| Chạy 4 server cùng lúc chậm | VRAM không đủ → chỉ nên chạy đúng **2 server của mode cần dùng** (`baseline` hoặc `turboquant`). |
| Log hiển thị `AUTO → ...` | Đang ở `--mode auto`, hệ thống tự quyết theo `.env`. Muốn cố định, truyền rõ `--mode baseline`/`--mode turboquant`. |
| Banner báo `GGUF Status: NOT FOUND` | File GGUF chưa tồn tại trong `models/`. Kiểm tra lại mục [§3.4](#34-tải-model-gguf). |

### Quy trình chuẩn khi gặp lỗi kết nối

1. Kiểm tra server còn sống: `curl -s http://127.0.0.1:<port>/health`.
2. Xem log server: `tail -n 50 logs/server_*.log`.
3. Đảm bảo `--mode` của `main.py` khớp với server đang chạy.
4. Khởi động lại server nếu cần (`kill $(cat logs/server_*.pid)` rồi start lại).

---

## Tài liệu tham khảo thêm

- `CLAUDE.md` — tổng quan kiến trúc, vai trò 5 agent và state management.
- `turboquant_integration_guide_v1.md`, `turboquant_integration_guide_v2.md` — chi tiết tích hợp TurboQuant.
- `docs/md/agents.md`, `docs/md/prompts.md` — mô tả từng agent và prompt.
