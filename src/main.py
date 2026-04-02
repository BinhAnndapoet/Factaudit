import os
from dotenv import load_dotenv
from main_graph import master_graph

def run_fact_audit():
    # 1. Khởi tạo dữ liệu mồi (Initial State)
    # Đây là dữ liệu tương đương với việc nạp file fact_cat.json trong code gốc
    initial_state = {
        "main_task": "Fact-Checking",
        "final_new_task": "Health and Medical Fake News", # Kịch bản đầu tiên muốn chạy
        "categories": {
            "Fact-Checking": [
                "Health and Medical Fake News", 
                "Political Rumors",
                "Financial Misinformation"
            ]
        },
        "taxonomy_scores": {
            "Health and Medical Fake News": 0.0,
        },
        "bad_cases_formatted": "",
        "is_terminated": False,
        "seed_data": [],
        "memory_pool": [] # Lưu ý: Tên này phải khớp với biến Reducer trong MainState của bạn
    }

    print("🚀 BẮT ĐẦU KÍCH HOẠT HỆ THỐNG FACT-AUDIT 🚀")
    print("-" * 60)

    # 2. Thiết lập Cấu hình (CỰC KỲ QUAN TRỌNG)
    # Vì Prober lặp 30 lần, mỗi vòng đi qua nhiều Node -> Tổng số lần nhảy node rất lớn.
    # Mặc định LangGraph ngắt ở 25 steps (chống lặp vô hạn). Ta phải nới lỏng ra 1000.
    config = {"recursion_limit": 1000}

    try:
        # 3. Dùng .stream() để xem dữ liệu chảy qua từng Node theo thời gian thực
        for event in master_graph.stream(initial_state, config=config):
            for node_name, node_state in event.items():
                print(f"\n✅ [Node Hoàn Thành]: {node_name.upper()}")
                
                # In log tùy chỉnh để theo dõi trạng thái
                if node_name == "inquirer_node":
                    seed_count = len(node_state.get('seed_data', []))
                    print(f"   -> Đã sinh {seed_count} seed cases. Đang đẩy vào Evaluation Subgraph...")
                
                elif node_name == "evaluation_subgraph":
                    # Lưu ý: Khi Map-Reduce (Send API) chạy, nó trả về list các kết quả
                    print(f"   -> Đã hoàn tất 1 luồng 'tra tấn' Target LLM.")
                
                elif node_name == "aggregate_bad_cases_node":
                    print(f"   -> Đã gom các Bad Cases xong. Chuyển cho Appraiser phân tích.")
                
                elif node_name == "appraiser_subgraph":
                    if node_state.get("is_terminated"):
                        print(f"   -> 🎯 Appraiser báo cáo Taxonomy đã hoàn hảo. DỪNG HỆ THỐNG!")
                    else:
                        print(f"   -> 🔄 Đã chốt kịch bản mới: {node_state.get('final_new_task')}")
                        print(f"   -> Bắt đầu vòng lặp tiến hóa tiếp theo...")

    except Exception as e:
        print(f"\n❌ LỖI TRONG QUÁ TRÌNH CHẠY: {e}")

if __name__ == "__main__":
    # Đảm bảo đã load API Key từ file .env
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Lỗi: Vui lòng set GEMINI_API_KEY trong file .env!")
    else:
        run_fact_audit()