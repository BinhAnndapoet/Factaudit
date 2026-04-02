"""
Target LLM Node

This module represents the model being audited. It receives the test case
and generates a response, which will later be evaluated against the 
Gold Reference Answer in Evaluator Phase 2.
"""

from langchain_core.prompts import PromptTemplate

# Import Thí sinh từ config
from config import llm_target

# Mượn lại Prompt và Schema cấu trúc của Evaluator để đảm bảo format đầu ra giống nhau
from evaluator.eval_prompts import gen_fact_problem_prompt
from evaluator.eval_state import ReferenceOutput

def _extract_prompt_data(current_case: dict) -> tuple:
    prompt_data = current_case.get("prompt", {})
    return (
        prompt_data.get("source_claim", ""),
        prompt_data.get("auxiliary_info", "")
    )

def _build_question_context(claim: str, aux_info: str) -> str:
    if aux_info and aux_info.strip():
        return f"Claim: {claim}\nContext: {aux_info}"
    return f"Claim: {claim}"


def target_llm_node(state: dict):
    """
    Hàm này đại diện cho Target LLM. 
    Chạy song song với Evaluator Phase 1 trong Main Graph.
    """
    print("\n[Target LLM] Thí sinh đang giải bài tập...")
    
    current_case = state.get("current_case", {})
    claim, aux_info = _extract_prompt_data(current_case)
    question_context = _build_question_context(claim, aux_info)
    
    # Ép Thí sinh cũng phải trả về đúng chuẩn [Verdict] + Justification
    # (Nếu Thí sinh là Black-box API không hỗ trợ Structured Output, 
    # ta sẽ phải dùng Regex để bóc tách text tĩnh ở đây)
    chain = PromptTemplate.from_template(gen_fact_problem_prompt) | llm_target.with_structured_output(ReferenceOutput)
    
    try:
        res = chain.invoke({"question": question_context})
        formatted_response = f"[{res.verdict}] {res.justification}"
        print(f"[Target LLM] Đã trả lời xong! Verdict: {res.verdict}")
    except Exception as e:
        print(f"[Target LLM] Thí sinh gặp lỗi (Timeout/Format): {e}")
        # Xử lý fallback nếu mô hình test bị lỗi
        formatted_response = "[Not Enough Information] Model failed to generate a valid response."

    # Trả về kết quả để cập nhật vào State tổng, chờ Phase 2 chấm điểm
    return {"target_response": formatted_response}