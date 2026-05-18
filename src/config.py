# =====================================================================
# Factaudit/src/config.py
# =====================================================================
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI  # <-- THÊM DÒNG NÀY

load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY must be set in .env file")

# Các Agent nội bộ (Explorer, Judge, Scorer) giữ nguyên cấu hình Ollama hoặc Gemini của bạn
llm_explorer = ChatOllama(model="llama3.1", temperature=1.0, format="json")
llm_judge = ChatOllama(model="llama3.1", temperature=0, format="json")
llm_scorer = ChatOllama(model="llama3.1", temperature=0, format="json")

# 4. Target LLM (Mô hình bị kiểm toán)
# THAY ĐỔI: Chuyển từ ChatOllama sang ChatOpenAI để kết nối với llama-server TurboQuant+
# THÊM DÒNG LOG NÀY ĐỂ XÁC NHẬN KHI STARTUP:
target_base_url = os.getenv("TARGET_LLM_BASE_URL", "http://localhost:8001/v1")
print(f"🚀 [HỆ THỐNG] Đang kết nối Thí sinh (Target LLM) tới hạ tầng tối ưu KV Cache [TurboQuant+] tại: {target_base_url}")

llm_target = ChatOpenAI(
    model="turboquant-target",
    temperature=0.6,
    base_url=os.getenv("TARGET_LLM_BASE_URL", "http://localhost:8001/v1"),
    api_key=os.getenv("TARGET_LLM_API_KEY", "not-needed")
)

# Global Constants
MAX_RETRIES = 3
MAX_WEB_CHECKS = 2
LOW_SCORE_THRESHOLD = 3.0
MAX_ITERATIONS = 3