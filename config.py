import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in .env file")


# 1. Explorer Agent (Appraiser, Prober, Evaluator Phase 1)
# Yêu cầu: Nhiệt độ cao (1.0) để tạo ra kịch bản mới lạ, khoét sâu điểm yếu.
llm_explorer = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=1.0,
    api_key=GEMINI_API_KEY
)


# 2. Judge Agent (Inquirer, Quality Inspector, Internal Judge)
# Yêu cầu: Nhiệt độ thấp (0.0) để đảm bảo tính công bằng, chính xác và không ảo giác.
# Lưu ý: Inquirer sử dụng cấu hình này để tạo dữ liệu seed có tính tái lập.
llm_judge = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.0,
    api_key=GEMINI_API_KEY
)

# 3. Scorer Agent (Evaluator Phase 2)
# Yêu cầu: Nhiệt độ (0.0), dùng mô hình nhẹ hơn để đánh giá quy mô lớn tiết kiệm chi phí.
llm_scorer = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-8b", 
    temperature=0.0,
    api_key=GEMINI_API_KEY
)


# Global Constants
MAX_RETRIES = 3              # Số lần lặp tối đa khi LLM bị Judge/Quality Inspector từ chối
LOW_SCORE_THRESHOLD = 3.0    # Ngưỡng điểm để xếp một test case vào loại "Bad Case" (dành cho Evaluator)
MAX_ITERATIONS = 30          # Số vòng lặp tối đa của Prober (Iterative Probing) theo bài báo