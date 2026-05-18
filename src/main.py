import os
import sys
import datetime
import re
import json
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

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


# ==========================================
# HÀM TÍNH TOÁN METRICS THEO BÀI BÁO FACT-AUDIT
# ==========================================
def calculate_factaudit_metrics(memory_pool: list):
    """
    Duyệt qua memory_pool để bóc tách nhãn kết luận [Verdict] và Điểm số,
    tính toán chính xác các chỉ số IMR, JFR, Grade và tỷ lệ lỗi JFR/IMR.
    """
    total_tests = len(memory_pool)
    if total_tests == 0:
        print("\n⚠️ [METRICS] Không có dữ liệu kiểm thử nào trong Memory Pool để tính toán chỉ số!")
        return None
        
    imr_count = 0
    jfr_count = 0
    total_grade = 0.0
    
    for record in memory_pool:
        # 1. Trích xuất Điểm số (Grade) - Hỗ trợ cả trường 'grade' và 'score' từ Evaluator Node
        grade = float(record.get("grade", record.get("score", 10.0)))
        total_grade += grade
        
        # 2. Trích xuất câu trả lời của Target LLM và Reference Answer để so sánh nhãn Verdict
        target_resp = record.get("target_response", "")
        refer_resp = record.get("refer_answer", "")
        
        # Sử dụng Regex bóc tách nhãn nằm trong ngoặc vuông (Ví dụ: [Factual], [Non-Factual])
        target_verdict_match = re.search(r'\[(.*?)\]', target_resp)
        refer_verdict_match = re.search(r'\[(.*?)\]', refer_resp)
        
        target_verdict = target_verdict_match.group(1).strip().lower() if target_verdict_match else "unknown_target"
        refer_verdict = refer_verdict_match.group(1).strip().lower() if refer_verdict_match else "unknown_refer"
        
        # 3. Phân loại và kiểm tra điều kiện lỗi nghiêm ngặt theo định nghĩa bài báo
        # Một case bị coi là tệ (is_bad_case) khi Grade <= 3.0 (Quy tắc ép điểm của bài báo)
        is_bad_case = grade <= 3.0
        # Kiểm tra xem mô hình mục tiêu có đoán trúng nhãn kết luận Verdict hay không
        is_correct_verdict = (target_verdict == refer_verdict)
        
        if is_bad_case:
            imr_count += 1
            # Định nghĩa JFR: Mô hình bị điểm kém (Grade <= 3) nhưng đoán ĐÚNG nhãn kết luận
            # Điều này chứng tỏ mô hình đoán mò đúng nhãn nhưng lập luận (Justification) sai/hỏng kiến thức
            if is_correct_verdict:
                jfr_count += 1
                
    # 4. Tính toán tỷ lệ phần trăm và các chỉ số trung bình
    imr_percentage = (imr_count / total_tests) * 100
    jfr_percentage = (jfr_count / total_tests) * 100
    avg_grade = total_grade / total_tests
    # Tỷ lệ ngụy biện bổ trợ (Phụ lục K): Có bao nhiêu % ca lỗi là do ngụy biện/lập luận hỏng
    jfr_imr_ratio = (jfr_count / imr_count) * 100 if imr_count > 0 else 0.0
    
    # 5. Xuất bảng báo cáo số liệu cực kỳ đẹp mắt ra Terminal và cả file log
    print("\n" + "="*60)
    print("📊 BÁO CÁO KIỂM TOÁN NĂNG LỰC REASONING (FACT-AUDIT METRICS)")
    print("="*60)
    print(f"🔹 Tổng số ca kiểm thử (Total Tests): {total_tests}")
    print(f"⭐ Điểm số trung bình (Average Grade): {avg_grade:.2f} / 10.0")
    print(f"🚨 Tỷ lệ lộ lỗ hổng hệ thống (IMR):   {imr_percentage:.2f}% ({imr_count}/{total_tests})")
    print(f"🕵️‍♂️ Tỷ lệ lỗi lập luận/ngụy biện (JFR):  {jfr_percentage:.2f}% ({jfr_count}/{total_tests})")
    print(f"⚖️  Tỷ lệ JFR/IMR trong nhóm ca lỗi:   {jfr_imr_ratio:.2f}%")
    print("="*60 + "\n")
    
    return {
        "IMR": imr_percentage,
        "JFR": jfr_percentage,
        "Avg_Grade": avg_grade,
        "JFR_IMR_Ratio": jfr_imr_ratio
    }


# ==========================================
# LUỒNG CHẠY CHÍNH CỦA HỆ THỐNG
# ==========================================
def main():
    # Khởi tạo logger để chia đôi dòng log (Vừa hiện terminal vừa lưu file)
    logger = DualLogger()
    sys.stdout = logger
    sys.stderr = logger

    print("==================================================")
    print("🚀 Khởi chạy Đồ thị Multi-Agent FACT-AUDIT Workflow")
    print("==================================================")
    
    # Khởi tạo trạng thái ban đầu của đồ thị tổng
    initial_state = {
        "fact_checking_objects": ["Complex Claim", "Fake News", "Social Rumor"],
        "memory_pool": []
    }
    
    # Biến cục bộ để backup thu hoạch memory_pool phòng trường hợp lỗi stream nửa chừng
    final_memory_pool = []

    try:
        # Chạy đồ thị dưới dạng stream cập nhật để bắt trạng thái các node thời gian thực
        for chunk in master_graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_state in chunk.items():
                
                # Liên tục thu hoạch và cập nhật memory_pool mới nhất từ state của graph
                if "memory_pool" in node_state and node_state["memory_pool"]:
                    final_memory_pool = node_state["memory_pool"]
                
                # In ra log trạng thái theo luồng xử lý đồ thị gốc của bạn
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
                        
        print("\n✅ Đồ thị LangGraph đã hoàn tất toàn bộ chu trình thực thi!")

    except Exception as e:
        print(f"\n❌ LỖI TRONG QUÁ TRÌNH CHẠY ĐỒ THỊ: {e}")

    # =====================================================================
    # ĐOẠN PHÂN TÍCH VÀ XUẤT BÁO CÁO METRICS CUỐI CÙNG
    # =====================================================================
    # Nếu kết thúc vòng lặp stream mà final_memory_pool vẫn trống, cố gắng rút trực tiếp từ state cuối
    if not final_memory_pool:
        try:
            graph_state = master_graph.get_state()
            final_memory_pool = graph_state.values.get("memory_pool", [])
        except:
            pass

    # Gọi hàm tính toán metrics để công bố bảng số liệu lên màn hình
    metrics_result = calculate_factaudit_metrics(final_memory_pool)
    
    # Xuất file JSON lưu trữ cấu trúc báo cáo tự động để lưu vết hoặc vẽ biểu đồ sau này
    if metrics_result:
        report_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "summary_metrics": metrics_result,
            "total_cases_audited": len(final_memory_pool),
            "detailed_memory_pool": final_memory_pool
        }
        report_filename = "factaudit_evaluation_report.json"
        with open(report_filename, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=4)
        print(f"💾 [HỆ THỐNG] Đã lưu trữ báo cáo chi tiết dạng cấu trúc JSON vào file: '{report_filename}'")


if __name__ == "__main__":
    # Nạp biến môi trường trước khi kích hoạt
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Lỗi: Vui lòng set GEMINI_API_KEY trong file .env!")
    else:
        main()