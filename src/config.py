import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in .env file")


# # 1. Explorer Agent (Appraiser, Prober, Evaluator Phase 1)
# # Yêu cầu: Nhiệt độ cao (1.0) để tạo ra kịch bản mới lạ, khoét sâu điểm yếu.
# llm_explorer = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", 
#     temperature=1.0,
#     api_key=GEMINI_API_KEY
# )


# # 2. Judge Agent (Inquirer, Quality Inspector, Internal Judge)
# # Yêu cầu: Nhiệt độ thấp (0.0) để đảm bảo tính công bằng, chính xác và không ảo giác.
# # Lưu ý: Inquirer sử dụng cấu hình này để tạo dữ liệu seed có tính tái lập.
# llm_judge = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", 
#     temperature=0.0,
#     api_key=GEMINI_API_KEY
# )

# # 3. Scorer Agent (Evaluator Phase 2)
# # Yêu cầu: Nhiệt độ (0.0), dùng mô hình nhẹ hơn để đánh giá quy mô lớn tiết kiệm chi phí.
# llm_scorer = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", 
#     temperature=0.0,
#     api_key=GEMINI_API_KEY
# )

# llm_target = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", # Ví dụ ta mang bản Pro ra test
#     temperature=0.6, 
#     api_key=GEMINI_API_KEY
# )

# 1. Explorer Agent (Appraiser, Prober, Evaluator Phase 1)
# Yêu cầu: Nhiệt độ cao (1.0) để tạo ra kịch bản mới lạ.
llm_explorer = ChatOllama(
    model="llama3.1", 
    temperature=1.0,
    format="json"
)

# 2. Judge Agent (Inquirer, Quality Inspector, Internal Judge)
# Yêu cầu: Nhiệt độ thấp (0.0) để đảm bảo tính chính xác.
llm_judge = ChatOllama(
    model="llama3.1", 
    temperature=0,
    format="json"
)

# 3. Scorer Agent (Evaluator Phase 2)
# Yêu cầu: Nhiệt độ (0.0). Có thể dùng model nhỏ hơn (ví dụ: phi3) để tăng tốc độ.
llm_scorer = ChatOllama(
    model="llama3.1", 
    temperature=0,
    format="json"
)

# 4. Target LLM (Mô hình bị kiểm toán)
# Có thể dùng Llama 2 để bám sát mã nguồn gốc của bài báo.
llm_target = ChatOllama(
    model="llama3.1", 
    temperature=0.6
)


# Global Constants
MAX_RETRIES = 3              # Số lần lặp tối đa khi LLM bị Judge/Quality Inspector từ chối
MAX_WEB_CHECKS = 2           # Số lần tối đa được phép quay lại web_check_node để sửa lỗi trước khi bỏ qua
LOW_SCORE_THRESHOLD = 3.0    # Ngưỡng điểm để xếp một test case vào loại "Bad Case" (dành cho Evaluator)
# MAX_ITERATIONS = 30          # Số vòng lặp tối đa của Prober (Iterative Probing) theo bài báo
MAX_ITERATIONS = 3