import re
from typing import List, Dict

def calculate_factaudit_metrics(memory_pool: List[Dict]):
    """
    Tính toán các chỉ số IMR, JFR và Grade từ danh sách kết quả kiểm toán.
    memory_pool: List các dictionary chứa trạng thái của từng case đã chạy qua Evaluator.
    """
    total_tests = len(memory_pool)
    if total_tests == 0:
        print("⚠️ Không có dữ liệu kiểm thử nào trong Memory Pool!")
        return None
        
    imr_count = 0
    jfr_count = 0
    total_grade = 0.0
    
    for record in memory_pool:
        # 1. Trích xuất Điểm số (Grade) - Mặc định là 10 nếu không có lỗi
        # Giả sử Output của Evaluator của bạn lưu điểm trong field "grade" hoặc "score"
        grade = float(record.get("grade", record.get("score", 10.0)))
        total_grade += grade
        
        # 2. Trích xuất câu trả lời của Target LLM và Reference Answer
        target_resp = record.get("target_response", "")
        refer_resp = record.get("refer_answer", "")
        
        # Dùng Regex bóc tách nhãn [Verdict] (VD: [Factual], [Non-Factual])
        target_verdict_match = re.search(r'\[(.*?)\]', target_resp)
        refer_verdict_match = re.search(r'\[(.*?)\]', refer_resp)
        
        target_verdict = target_verdict_match.group(1).strip().lower() if target_verdict_match else "unknown_target"
        refer_verdict = refer_verdict_match.group(1).strip().lower() if refer_verdict_match else "unknown_refer"
        
        # 3. Kiểm tra các điều kiện theo định nghĩa của bài báo FACT-AUDIT
        # is_bad_case: Điểm số <= 3.0 (Quy tắc ép điểm của FACT-AUDIT)
        is_bad_case = grade <= 3.0
        
        # is_correct_verdict: Mô hình mục tiêu đoán đúng nhãn Verdict
        is_correct_verdict = (target_verdict == refer_verdict)
        
        if is_bad_case:
            imr_count += 1
            # Nếu bị coi là lỗi (Grade <= 3) MÀ nhãn vẫn đúng -> Lỗi do phần giải thích (Justification)
            if is_correct_verdict:
                jfr_count += 1
                
    # 4. Tính toán tỷ lệ phần trăm
    imr_percentage = (imr_count / total_tests) * 100
    jfr_percentage = (jfr_count / total_tests) * 100
    avg_grade = total_grade / total_tests
    
    # Tính tỷ lệ bổ trợ JFR/IMR (Phụ lục K)
    jfr_imr_ratio = (jfr_count / imr_count) * 100 if imr_count > 0 else 0.0
    
    # 5. In Bảng báo cáo
    print("\n" + "="*60)
    print("📊 BÁO CÁO KIỂM TOÁN FACT-AUDIT (METRICS REPORT)")
    print("="*60)
    print(f"Tổng số ca kiểm thử (Total Tests): {total_tests}")
    print(f"⭐ Điểm trung bình (Average Grade): {avg_grade:.2f} / 10.0")
    print(f"🚨 Tỷ lệ lộ lỗ hổng (IMR):         {imr_percentage:.2f}% ({imr_count}/{total_tests})")
    print(f"🕵️  Tỷ lệ lỗi lập luận (JFR):        {jfr_percentage:.2f}% ({jfr_count}/{total_tests})")
    print(f"⚖️  Tỷ lệ JFR/IMR (Ngụy biện/Tổng lỗi): {jfr_imr_ratio:.2f}%")
    print("="*60)
    
    return {
        "IMR": imr_percentage,
        "JFR": jfr_percentage,
        "Avg_Grade": avg_grade,
        "JFR_IMR_Ratio": jfr_imr_ratio
    }