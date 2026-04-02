import os
import sys
import datetime
from dotenv import load_dotenv
from main_graph import master_graph

# ==========================================
# CẤU HÌNH BẮT LOG TỪ TERMINAL
# ==========================================
class DualLogger:
    """
    Class này hoạt động như một bộ chia (Tee):
    Mỗi khi hệ thống gọi lệnh print(), nó sẽ in ra màn hình (terminal)
    đồng thời ghi trực tiếp xuống file log.
    """
    def __init__(self, log_dir="logs"):
        # Tự động tạo thư mục logs nếu chưa có
        os.makedirs(log_dir, exist_ok=True)
        
        # Tạo tên file log dán nhãn theo thời gian chạy
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_filepath = os.path.join(log_dir, f"fact_audit_run_{timestamp}.log")
        
        self.terminal = sys.stdout
        self.log_file = open(self.log_filepath, "a", encoding="utf-8")
        
        # In thông báo để biết log đang được ghi ở đâu (chỉ in ra terminal)
        self.terminal.write(f"📁 Log của phiên chạy này đang được ghi tại: {self.log_filepath}\n")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush() # Ép ghi ngay xuống ổ cứng để không mất data nếu app bị crash

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


# ==========================================
# LOGIC CHÍNH CỦA CHƯƠNG TRÌNH
# ==========================================
def run_fact_audit():
    initial_state = {
        "main_task": "Fact-Checking",
        "final_new_task": "Health and Medical Fake News", 
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
        "memory_pool": []
    }

    print("🚀 BẮT ĐẦU KÍCH HOẠT HỆ THỐNG FACT-AUDIT 🚀")
    print("-" * 60)

    config = {"recursion_limit": 1000}

    try:
        for event in master_graph.stream(initial_state, config=config):
            for node_name, node_state in event.items():
                print(f"\n✅ [Node Hoàn Thành]: {node_name.upper()}")
                
                if node_name == "inquirer_node":
                    seed_count = len(node_state.get('seed_data', []))
                    print(f"   -> Đã sinh {seed_count} seed cases. Đang đẩy vào Evaluation Subgraph...")
                
                elif node_name == "evaluation_subgraph":
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
    # Nạp biến môi trường trước
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Lỗi: Vui lòng set GEMINI_API_KEY trong file .env!")
    else:
        # Bắt đầu ghi đè stdout (các lệnh print) và stderr (các lỗi crash đỏ)
        logger = DualLogger(log_dir="logs")
        sys.stdout = logger
        sys.stderr = logger 
        
        # Chạy chương trình
        run_fact_audit()