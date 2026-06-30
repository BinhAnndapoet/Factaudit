"""
===============================================
FACT-AUDIT Configuration Module  (v2 - No Ollama)
===============================================
Module này quản lý việc khởi tạo LLM instances và hỗ trợ
chuyển đổi (switching) giữa 2 chế độ:
- Baseline Mode: Không có TurboQuant (f32 cache)
- TurboQuant+ Mode: Có KV Cache Compression (turbo3/turbo4)

THAY ĐỔI LỚN (v2):
- Đã LOẠ BỎ HOÀN TOÀN dependency Ollama (langchain_ollama / ChatOllama
  / ENABLE_OLLAMA_FALLBACK / OLLAMA_*).
- Model giờ được tải trực tiếp về máy dưới dạng file GGUF trong thư mục
  Factaudit/models/ (ví dụ: models/Qwen3-14B-Q8_0.gguf) và được serve bởi
  server `llama-cpp-turboquant` qua giao thức OpenAI-compatible.
- Client (Factaudit) CHỈ giao tiếp với server qua REST API (ChatOpenAI);
  toàn bộ logic load/nén KV cache nằm trong server, không còn trong Factaudit.

Cấu trúc:
- LLMFactory: Factory pattern tạo LLM instances dựa trên mode (baseline|turboquant)
- Global Constants: Các hằng số sử dụng trong hệ thống
"""

import os
from pathlib import Path
from typing import Optional, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  # Fallback cloud (tuỳ chọn)

# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================
load_dotenv()

# ==========================================
# PATH CONSTANTS
# ==========================================
# config.py nằm tại: Factaudit/src/config.py
#   -> project root = parent của src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Thư mục chứa các file model GGUF tải trực tiếp về máy
MODELS_DIR = PROJECT_ROOT / "models"
# Tên file GGUF mặc định (có thể override qua env GGUF_MODEL_FILE / GGUF_MODEL_PATH)
DEFAULT_GGUF_FILE = "Qwen3-14B-Q8_0.gguf"


def resolve_gguf_model_path() -> Path:
    """
    Resolve đường dẫn tuyệt đối của file model GGUF trong Factaudit/models/.

    Phục vụ mục đích LOG/CẢNH BÁO cho người dùng biết file nào đang được
    server sử dụng. Client không trực tiếp load file này (server đã load rồi).

    Priority (cao -> thấp):
    1. GGUF_MODEL_PATH  : đường dẫn tuyệt đối/relative chỉ định trực tiếp
    2. GGUF_MODEL_FILE  : tên file .gguf nằm trong MODELS_DIR
    3. Scan MODELS_DIR   : file *.gguf đầu tiên tìm được (sắp xếp theo tên)
    4. Fallback          : DEFAULT_GGUF_FILE trong MODELS_DIR

    Returns:
        Path tới file GGUF (path luôn trả về, dù file có thể chưa tồn tại).
    """
    # 1. Đường dẫn trực tiếp (ưu tiên cao nhất)
    explicit = os.getenv("GGUF_MODEL_PATH")
    if explicit:
        return Path(explicit).expanduser()

    # 2. Tên file nằm trong thư mục models/
    model_file = os.getenv("GGUF_MODEL_FILE")
    if model_file:
        return (MODELS_DIR / model_file).resolve()

    # 3. Scan thư mục models/ lấy file .gguf đầu tiên
    if MODELS_DIR.is_dir():
        gguf_files = sorted(MODELS_DIR.glob("*.gguf"))
        if gguf_files:
            return gguf_files[0].resolve()

    # 4. Fallback mặc định
    return (MODELS_DIR / DEFAULT_GGUF_FILE).resolve()


# ==========================================
# LLM FACTORY CLASS
# ==========================================
class LLMFactory:
    """
    Factory Pattern để tạo LLM instances với mode switching.

    Hỗ trợ 2 chế độ:
    - "baseline":   Trỏ tới Baseline Server   (port 8080, cache f32)
    - "turboquant": Trỏ tới TurboQuant+ Server (port 8081, cache turbo3/turbo4)

    Cả 2 chế độ đều tạo ra ChatOpenAI instance trỏ tới BASELINE_API_BASE
    hoặc TURBOQUANT_API_BASE (theo .env). Không còn fallback Ollama.

    Usage:
        factory = LLMFactory(mode="turboquant")
        llm_explorer = factory.create_explorer()
        llm_judge = factory.create_judge()
    """

    def __init__(self, mode: Optional[Literal["baseline", "turboquant", "auto"]] = None):
        """
        Khởi tạo LLMFactory.

        Args:
            mode: Chế độ LLM inference
                - "baseline":   Force dùng Baseline mode
                - "turboquant": Force dùng TurboQuant+ mode
                - "auto":       Tự động quyết định dựa trên USE_TURBOQUANT từ .env
                - None:         Giống như "auto"
        """
        self._mode = self._determine_mode(mode)
        self._api_base = self._get_api_base()
        # MODEL_NAME chỉ là nhãn gửi trong field `model` của request (informational
        # với llama.cpp server). Nên đặt bằng alias của server (thường là tên file
        # GGUF không kèm đuôi) để /v1/models khớp.
        self._model_name = os.getenv("MODEL_NAME", "Qwen3-14B-Q8_0")
        self._api_key = os.getenv("API_KEY", "sk-not-required")
        self._timeout = int(os.getenv("TIMEOUT", "300"))
        # Đường dẫn GGUF để log/thông báo cho người dùng
        self._gguf_path = resolve_gguf_model_path()

        # Print mode info
        self._print_mode_info()

    def _determine_mode(self, mode: Optional[str]) -> str:
        """
        Xác định mode thực tế sẽ sử dụng.

        Priority (highest to lowest):
        1. Explicit mode parameter ("baseline" hoặc "turboquant")
        2. MODE environment variable
        3. USE_TURBOQUANT flag (nếu MODE=auto)
        """
        # 1. Explicit parameter has highest priority
        if mode in ["baseline", "turboquant"]:
            return mode

        # 2. Check MODE environment variable
        env_mode = os.getenv("MODE", "auto").lower()
        if env_mode in ["baseline", "turboquant"]:
            return env_mode

        # 3. Auto mode: check USE_TURBOQUANT flag
        use_turbo = os.getenv("USE_TURBOQUANT", "false").lower() == "true"
        return "turboquant" if use_turbo else "baseline"

    def _get_api_base(self) -> str:
        """
        Lấy API endpoint dựa trên mode đã xác định.

        - turboquant -> TURBOQUANT_API_BASE (mặc định http://localhost:8081/v1)
        - baseline   -> BASELINE_API_BASE   (mặc định http://localhost:8080/v1)
        """
        if self._mode == "turboquant":
            api_base = os.getenv("TURBOQUANT_API_BASE", "http://localhost:8081/v1")
        else:
            api_base = os.getenv("BASELINE_API_BASE", "http://localhost:8080/v1")

        return api_base

    @staticmethod
    def _truncate(text: str, width: int = 50) -> str:
        """Cắt chuỗi dài cho vừa ô hiển thị trong terminal."""
        text = str(text)
        if len(text) <= width:
            return text
        return "..." + text[-(width - 3):]

    def _print_mode_info(self):
        """In thông tin mode + đường dẫn file GGUF ra terminal để user tracking."""
        mode_display = "TurboQuant+ (KV Cache Compression)" if self._mode == "turboquant" else "Baseline (f32 cache)"

        # Kiểm tra file GGUF có tồn tại không (chỉ để cảnh báo, không block)
        if self._gguf_path.exists():
            gguf_status = f"found ({self._gguf_path.stat().st_size / (1024 ** 3):.1f} GB)"
        else:
            gguf_status = "NOT FOUND - server có thể không load được"

        print(f"┌" + "─" * 70 + "┐")
        print(f"│ {'LLM FACTORY INITIALIZED':^66} │")
        print(f"├" + "─" * 70 + "┤")
        print(f"│ Mode:        {self._truncate(mode_display):<50} │")
        print(f"│ API Base:    {self._truncate(self._api_base):<50} │")
        print(f"│ Model Alias: {self._truncate(self._model_name):<50} │")
        print(f"│ GGUF File:   {self._truncate(self._gguf_path.name):<50} │")
        print(f"│ GGUF Path:   {self._truncate(self._gguf_path, 50):<50} │")
        print(f"│ GGUF Status: {self._truncate(gguf_status):<50} │")

        # Print context size based on mode
        if self._mode == "turboquant":
            ctx_size = os.getenv("TURBOQUANT_CONTEXT_SIZE", "32768")
            print(f"│ Max Context: {self._truncate(ctx_size + ' tokens (4x capacity)'):<50} │")
        else:
            ctx_size = os.getenv("MAX_CONTEXT_SIZE", "8192")
            print(f"│ Max Context: {self._truncate(ctx_size + ' tokens'):<50} │")

        print(f"└" + "─" * 70 + "┘")

    @property
    def mode(self) -> str:
        """Get current mode."""
        return self._mode

    @property
    def api_base(self) -> str:
        """Get current API endpoint."""
        return self._api_base

    @property
    def gguf_path(self) -> Path:
        """Get resolved GGUF model file path (for logging/display)."""
        return self._gguf_path

    def switch_mode(self, new_mode: Literal["baseline", "turboquant"]) -> None:
        """
        Switch runtime mode và update API endpoint.

        Args:
            new_mode: "baseline" hoặc "turboquant"
        """
        old_mode = self._mode
        self._mode = new_mode
        self._api_base = self._get_api_base()

        print(f"\n🔄 [LLMFactory] Mode switched: {old_mode.upper()} → {new_mode.upper()}")
        print(f"   New API Base: {self._api_base}\n")

    # ==========================================
    # LLM CREATION METHODS
    # ==========================================

    def _create_base_llm(
        self,
        temperature: float,
        max_tokens: Optional[int] = None,
        format: Optional[str] = None
    ) -> ChatOpenAI:
        """
        Tạo base ChatOpenAI instance với cấu hình chung.

        Instance này trỏ tới API endpoint (BASELINE_API_BASE hoặc
        TURBOQUANT_API_BASE) tuỳ theo mode hiện tại của factory.

        Args:
            temperature: Temperature cho generation
            max_tokens: Maximum tokens (optional)
            format: Output format ("json" hoặc None)
                - Được GIỮ LẠI để tương thích ngược với signature của các agent.
                - Việc ép JSON output thực tế được xử lý qua .with_structured_output()
                  trong code agent; KHÔNG truyền response_format vào đây để tránh
                  xung đột với structured output (tool calling) của OpenAI-compatible API.

        Returns:
            ChatOpenAI instance
        """
        # Parse max_tokens
        if max_tokens is None:
            max_tokens = int(os.getenv("MAX_TOKENS", "4096"))

        # Build kwargs
        kwargs = {
            "model": self._model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": self._api_key,
            "timeout": self._timeout,
        }

        # NOTE: giữ tham số `format` để không phá vỡ contract `create_judge(format="json")`.
        # JSON mode được xử lý qua with_structured_output() ở tầng agent, không qua đây.
        # (Truyền response_format={"type":"json_object"} có thể gây xung đột với tool
        #  calling mà with_structured_output() sử dụng -> cố tình bỏ qua.)
        _ = format

        return ChatOpenAI(base_url=self._api_base, **kwargs)

    def _create_fallback_gemini(
        self,
        temperature: float,
        model: str = "gemini-2.5-flash"
    ) -> ChatGoogleGenerativeAI:
        """
        Tạo fallback Gemini instance (CLOUD, tuỳ chọn).

        CHỈ dùng khi có GEMINI_API_KEY và không muốn dùng server local.
        Không liên quan tới luồng GGUF/llama.cpp. Bỏ qua method này nếu
        không có key.

        Args:
            temperature: Temperature cho generation
            model: Gemini model name

        Returns:
            ChatGoogleGenerativeAI instance

        Raises:
            ValueError: nếu thiếu GEMINI_API_KEY
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in .env for Gemini fallback")

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )

    # ------------------------------------------
    # EXPLORER LLM (Appraiser, Prober, Evaluator Phase 1)
    # ------------------------------------------
    def create_explorer(self) -> ChatOpenAI:
        """
        Tạo Explorer LLM.

        Usage: Appraiser, Prober, Evaluator Phase 1
        Yêu cầu: Nhiệt độ cao (1.0) để tạo kịch bản mới lạ, sáng tạo
        """
        temp = float(os.getenv("TEMPERATURE_EXPLORER", "1.0"))

        llm = self._create_base_llm(temperature=temp)

        print(f"  ✓ llm_explorer created (temp={temp})")

        return llm

    # ------------------------------------------
    # JUDGE LLM (Inquirer, Quality Inspector)
    # ------------------------------------------
    def create_judge(self, format: Optional[str] = "json") -> ChatOpenAI:
        """
        Tạo Judge LLM.

        Usage: Inquirer, Quality Inspector, Internal Judge
        Yêu cầu: Nhiệt độ thấp (0.0) để đảm bảo tính chính xác, công bằng

        Args:
            format: "json" (mặc định) - giữ lại cho tương thích; JSON mode
                thực sự xử lý qua with_structured_output() ở tầng agent.
        """
        temp = float(os.getenv("TEMPERATURE_JUDGE", "0.0"))

        llm = self._create_base_llm(temperature=temp, format=format)

        print(f"  ✓ llm_judge created (temp={temp}, format={format})")

        return llm

    # ------------------------------------------
    # SCORER LLM (Evaluator Phase 2)
    # ------------------------------------------
    def create_scorer(self) -> ChatOpenAI:
        """
        Tạo Scorer LLM.

        Usage: Evaluator Phase 2
        Yêu cầu: Nhiệt độ thấp (0.0) để đánh giá nhất quán
        """
        temp = float(os.getenv("TEMPERATURE_SCORER", "0.0"))

        llm = self._create_base_llm(temperature=temp)

        print(f"  ✓ llm_scorer created (temp={temp})")

        return llm

    # ------------------------------------------
    # TARGET LLM (Model Under Test)
    # ------------------------------------------
    def create_target(self) -> ChatOpenAI:
        """
        Tạo Target LLM.

        Usage: Mô hình bị kiểm toán
        Yêu cầu: Nhiệt độ trung bình (0.6) để simulate user behavior
        """
        temp = float(os.getenv("TEMPERATURE_TARGET", "0.6"))

        llm = self._create_base_llm(temperature=temp)

        print(f"  ✓ llm_target created (temp={temp})")

        return llm

    # ------------------------------------------
    # BATCH CREATION (Create all LLMs at once)
    # ------------------------------------------
    def create_all(self) -> dict:
        """
        Tạo tất cả LLM instances và return dưới dạng dict.

        Returns:
            dict với keys: 'explorer', 'judge', 'scorer', 'target'
        """
        print(f"\n[LLMFactory] Creating all LLM instances...")

        return {
            "explorer": self.create_explorer(),
            "judge": self.create_judge(),
            "scorer": self.create_scorer(),
            "target": self.create_target()
        }


# ==========================================
# GLOBAL LLM INSTANCES (Lazy Initialization)
# ==========================================
"""
Sử dụng pattern: Lazy Initialization + Singleton

Các LLM instances sẽ được tạo khi first access,
dựa trên mode được xác định tại thời điểm đó.

Để switch mode runtime:
1. Gọi LLMFactory(mode="new_mode") để tạo factory mới
2. Re-assign các global LLM instances
"""

# Global factory instance (lazy loaded)
_llm_factory: Optional[LLMFactory] = None

# Global LLM instances (lazy loaded)
llm_explorer = None
llm_judge = None
llm_scorer = None
llm_target = None


def get_factory(mode: Optional[str] = None) -> LLMFactory:
    """
    Get singleton LLMFactory instance.

    Args:
        mode: Mode override (optional)

    Returns:
        LLMFactory instance
    """
    global _llm_factory

    if _llm_factory is None or (mode is not None and _llm_factory.mode != mode):
        _llm_factory = LLMFactory(mode=mode)

    return _llm_factory


def initialize_llms(mode: Optional[str] = None) -> dict:
    """
    Initialize tất cả LLM instances với mode specified.

    Args:
        mode: "baseline", "turboquant", hoặc None (use .env config)

    Returns:
        dict với các LLM instances
    """
    global llm_explorer, llm_judge, llm_scorer, llm_target

    factory = get_factory(mode=mode)
    llms = factory.create_all()

    # Update global instances
    llm_explorer = llms["explorer"]
    llm_judge = llms["judge"]
    llm_scorer = llms["scorer"]
    llm_target = llms["target"]

    return llms


def switch_llm_mode(new_mode: Literal["baseline", "turboquant"]) -> None:
    """
    Switch runtime mode và reinitialize tất cả LLM instances.

    Args:
        new_mode: "baseline" hoặc "turboquant"
    """
    global _llm_factory

    print(f"\n{'='*70}")
    print(f"🔄 SWITCHING LLM MODE TO: {new_mode.upper()}")
    print(f"{'='*70}\n")

    # Reset factory
    _llm_factory = None

    # Reinitialize LLMs
    initialize_llms(mode=new_mode)

    print(f"\n✅ All LLM instances switched to {new_mode.upper()} mode!\n")


# ==========================================
# GLOBAL CONSTANTS (From Original Config)
# ==========================================

# Retry & Threshold Settings
MAX_RETRIES = 3              # Số lần lặp tối đa khi LLM bị Judge/Quality Inspector từ chối
MAX_WEB_CHECKS = 2           # Số lần tối đa được phép quay lại web_check_node để sửa lỗi trước khi bỏ qua
LOW_SCORE_THRESHOLD = 3.0    # Ngưỡng điểm để xếp một test case vào loại "Bad Case" (dành cho Evaluator)
MAX_ITERATIONS = 3           # Số vòng lặp tối đa của Prober (Iterative Probing) theo bài báo

# Context Size Settings (dynamically based on mode)
def get_max_context_size() -> int:
    """Lấy max context size dựa trên current mode."""
    factory = get_factory()
    if factory.mode == "turboquant":
        return int(os.getenv("TURBOQUANT_CONTEXT_SIZE", "32768"))
    else:
        return int(os.getenv("MAX_CONTEXT_SIZE", "8192"))


# ==========================================
# INITIALIZATION (EXPLICIT — KHÔNG auto-init khi import)
# ==========================================
# TRƯỚC ĐÂY: module tự gọi initialize_llms() ngay khi import (mode=auto ->
# baseline do USE_TURBOQUANT=false). Điều này gây bug "tham chiếu cũ":
#   - Các agent viết `from config import llm_judge` -> chụp object BASELINE
#     (port 8080) ngay tại lúc import.
#   - Khi main.py SAU ĐÓ gọi initialize_llms(mode="turboquant") gán lại biến
#     global config.llm_judge, các agent vẫn giữ object CŨ -> turboquant vô
#     tình gọi server baseline 8080 (không chạy) -> "Connection error.".
#
# GIỜ: bỏ auto-init. Việc khởi tạo LLM phải tường minh qua:
#   - main.py     : initialize_llms(mode=...)  (chạy TRƯỚC khi stream graph)
#   - runtime     : switch_llm_mode(new_mode)
# Các agent đọc LLM tại call-time qua `config.llm_*` (đã sửa trong src/*/),
# nên luôn thấy instance đúng với mode hiện hành, bất kể thứ tự import.
#
# LƯU Ý: sau khi bỏ auto-init, các global llm_explorer/llm_judge/llm_scorer/
# llm_target là None cho tới khi có lời gọi initialize_llms() / switch_llm_mode().
