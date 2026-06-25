"""
===============================================
FACT-AUDIT Configuration Module
===============================================
Module này quản lý việc khởi tạo LLM instances và hỗ trợ
chuyển đổi (switching) giữa 2 chế độ:
- Baseline Mode: Không có TurboQuant (Q8_0/FP16)
- TurboQuant+ Mode: Có KV Cache Compression (turbo3/turbo4)

Cấu trúc:
- LLMFactory: Factory pattern tạo LLM instances dựa trên mode
- Global Constants: Các hằng số sử dụng trong hệ thống
"""

import os
from typing import Optional, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================
load_dotenv()

# ==========================================
# LLM FACTORY CLASS
# ==========================================
class LLMFactory:
    """
    Factory Pattern để tạo LLM instances với mode switching.

    Hỗ trợ 2 chế độ:
    - "baseline": Sử dụng API endpoint của Baseline Server (port 8080)
    - "turboquant": Sử dụng API endpoint của TurboQuant+ Server (port 8081)

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
                - "baseline": Force dùng Baseline mode
                - "turboquant": Force dùng TurboQuant+ mode
                - "auto": Tự động quyết định dựa trên USE_TURBOQUANT từ .env
                - None: Giống như "auto"
        """
        self._mode = self._determine_mode(mode)
        self._api_base = self._get_api_base()
        self._model_name = os.getenv("MODEL_NAME", "Llama-3-8B-Instruct")
        self._api_key = os.getenv("API_KEY", "sk-not-required")
        self._timeout = int(os.getenv("TIMEOUT", "300"))
        self._enable_fallback = os.getenv("ENABLE_OLLAMA_FALLBACK", "true").lower() == "true"

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
        """
        if self._mode == "turboquant":
            api_base = os.getenv("TURBOQUANT_API_BASE", "http://localhost:8081/v1")
        else:
            api_base = os.getenv("BASELINE_API_BASE", "http://localhost:8080/v1")

        return api_base

    def _print_mode_info(self):
        """In thông tin mode ra terminal để user tracking."""
        mode_display = "TurboQuant+ (KV Cache Compression)" if self._mode == "turboquant" else "Baseline (Q8_0/FP16)"

        print(f"┌" + "─" * 70 + "┐")
        print(f"│ {'LLM FACTORY INITIALIZED':^66} │")
        print(f"├" + "─" * 70 + "┤")
        print(f"│ Mode:        {mode_display:<50} │")
        print(f"│ API Base:    {self._api_base:<50} │")
        print(f"│ Model:       {self._model_name:<50} │")

        # Print context size based on mode
        if self._mode == "turboquant":
            ctx_size = os.getenv("TURBOQUANT_CONTEXT_SIZE", "32768")
            print(f"│ Max Context: {ctx_size + ' tokens (4x capacity)':<50} │")
        else:
            ctx_size = os.getenv("MAX_CONTEXT_SIZE", "8192")
            print(f"│ Max Context: {ctx_size + ' tokens':<50} │")

        print(f"└" + "─" * 70 + "┘")

    @property
    def mode(self) -> str:
        """Get current mode."""
        return self._mode

    @property
    def api_base(self) -> str:
        """Get current API endpoint."""
        return self._api_base

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

        Args:
            temperature: Temperature cho generation
            max_tokens: Maximum tokens (optional)
            format: Output format ("json" hoặc None)

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

        # Add format if specified (for structured output)
        if format == "json":
            # Note: ChatOpenAI supports response_format parameter
            # but we'll handle JSON via with_structured_output() in agent code
            pass

        return ChatOpenAI(base_url=self._api_base, **kwargs)

    def _create_fallback_ollama(
        self,
        temperature: float,
        model: Optional[str] = None,
        format: Optional[str] = None
    ) -> ChatOllama:
        """
        Tạo fallback ChatOllama instance nếu llama-cpp-turboquant unavailable.

        Args:
            temperature: Temperature cho generation
            model: Ollama model name (default from .env)
            format: Output format ("json" hoặc None)

        Returns:
            ChatOllama instance
        """
        if model is None:
            model = os.getenv("OLLAMA_MODEL", "llama3.1")

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        return ChatOllama(
            model=model,
            temperature=temperature,
            format=format,
            base_url=base_url
        )

    def _create_fallback_gemini(
        self,
        temperature: float,
        model: str = "gemini-2.5-flash"
    ) -> ChatGoogleGenerativeAI:
        """
        Tạo fallback Gemini instance (nếu có API key).

        Args:
            temperature: Temperature cho generation
            model: Gemini model name

        Returns:
            ChatGoogleGenerativeAI instance
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
        """
        temp = float(os.getenv("TEMPERATURE_JUDGE", "0.0"))

        llm = self._create_base_llm(temperature=temp)

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
# AUTO-INITIALIZATION ON MODULE IMPORT
# ==========================================

# Khi module được import, tự động initialize LLMs
# dựa trên config từ .env (hoặc mode được passed từ main.py)
_initialized = False


def ensure_initialized(mode: Optional[str] = None):
    """Ensure LLM instances are initialized."""
    global _initialized

    if not _initialized or (mode is not None and get_factory().mode != mode):
        initialize_llms(mode=mode)
        _initialized = True


# Auto-initialize on first import (sẽ được override bởi main.py nếu có --mode argument)
if not _initialized:
    initialize_llms()
    _initialized = True
