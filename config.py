import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in .env file")


# LLM Configurations (Mô hình ngôn ngữ)
# Tác tử sáng tạo (Appraiser, Inquirer, Prober) cần temperature cao để đa dạng
creative_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=1.0,
    api_key=GEMINI_API_KEY
)

# Tác tử phân tích/đánh giá (Quality Inspector, Judge, Evaluator) cần logic chặt chẽ, nhiệt độ thấp
analytical_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.0,
    api_key=GEMINI_API_KEY
)

# Mô hình giá rẻ/nhanh để xử lý các tác vụ phân loại nhẹ nhàng (tùy chọn)
mini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.0,
    api_key=GEMINI_API_KEY
)


# Global Constants (Hằng số hệ thống)
MAX_RETRIES = 3              # Số lần lặp tối đa khi LLM bị Judge/Quality Inspector từ chối
LOW_SCORE_THRESHOLD = 3.0    # Ngưỡng điểm để xếp một test case vào loại "Bad Case" (dành cho Evaluator)
MAX_ITERATIONS = 30          # Số vòng lặp tối đa của Prober (Iterative Probing) theo bài báo