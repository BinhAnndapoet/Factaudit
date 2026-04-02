"""
Master Workflow for FACT-AUDIT

This module connects all agents (Appraiser, Inquirer, Quality Inspector, 
Target LLM, Evaluator, and Prober) into a fully automated, nested StateGraph.
It utilizes LangGraph's Send API for parallel map-reduce (Fan-out/Fan-in) 
during the evaluation phase.
"""

import operator
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# ==== IMPORT GLOBAL CONFIG ====
from config import MAX_ITERATIONS, MAX_WEB_CHECKS

# ==== IMPORT CÁC ĐỒ THỊ CON & NODES ĐÃ LÀM ====
# 1. Appraiser
from appraiser.appraiser_agent import appraiser_graph
# 2. Inquirer
from inquirer.inquirer_agent import generate_seed_node
# 3. Quality Inspector (Lấy các node kiểm duyệt thô & tinh)
from quality_inspector.inspector_agent import web_check_node, llm_inspection_node
# 4. Target LLM & Evaluator Phase 1, Phase 2
from target_model.target_agent import target_llm_node
from evaluator.eval_agent import evaluator_phase1_graph, evaluator_phase2_score_node
# 5. Prober
from prober.prober_agent import prober_node


# ==========================================
# PHẦN 1: EVALUATION SUB-GRAPH (VÒNG LẶP TRA TẤN)
# ==========================================

class EvaluationState(TypedDict):
    """State riêng cho mỗi luồng Evaluation (chạy độc lập 10 luồng)."""
    task_name: str
    current_case: dict
    target_response: str
    reference_answer: str
    score: float
    comparison: str
    iteration_count: int
    # Reducer operator.add giúp cộng dồn lịch sử sau mỗi vòng lặp
    memory_pool: Annotated[List[Dict], operator.add] 
    retry_count: int

def save_memory_node(state: EvaluationState):
    """Gói kết quả chấm điểm và lưu vào Memory Pool."""
    record = {
        "prompt": state["current_case"].get("prompt", {}),
        "test_mode": state["current_case"].get("test_mode", ""),
        "key_point": state["current_case"].get("key_point", ""),
        "target_response": state.get("target_response", ""),
        "reference_answer": state.get("reference_answer", ""),
        "score": state.get("score", 0.0),
        "comparison": state.get("comparison", "")
    }
    return {"memory_pool": [record]}

def route_after_inspection(state: EvaluationState):
    current_case = state.get("current_case", {})
    
    if current_case.get("web_error") or current_case.get("llm_error"):
        # Nếu đã thử sửa quá MAX_WEB_CHECKS lần mà vẫn lỗi, ép nó bỏ qua web_check 
        # và đẩy một flag lỗi vào Target LLM, hoặc dùng case hiện tại luôn.
        if state.get("retry_count", 0) >= MAX_WEB_CHECKS:
            print("[Warning] Quá số lần retry ở Inspector. Cho qua với lỗi bảo lưu.")
            return ["target_llm_node", "evaluator_phase1_subgraph"]
        return "web_check_node" 
    
    return ["target_llm_node", "evaluator_phase1_subgraph"]

def route_prober_loop(state: EvaluationState):
    """Kiểm tra điều kiện Iterative Probing."""
    if state["iteration_count"] < MAX_ITERATIONS:
        return "prober_node"
    return END

# --- Build Evaluation Sub-graph ---
eval_builder = StateGraph(EvaluationState)

eval_builder.add_node("web_check_node", web_check_node)
eval_builder.add_node("llm_inspection_node", llm_inspection_node)
eval_builder.add_node("target_llm_node", target_llm_node)
eval_builder.add_node("evaluator_phase1_subgraph", evaluator_phase1_graph)
eval_builder.add_node("evaluator_phase2_score_node", evaluator_phase2_score_node)
eval_builder.add_node("save_memory_node", save_memory_node)
eval_builder.add_node("prober_node", prober_node)

# Luồng chạy Evaluation
eval_builder.add_edge(START, "web_check_node")
eval_builder.add_edge("web_check_node", "llm_inspection_node")
eval_builder.add_conditional_edges(
    "llm_inspection_node", 
    route_after_inspection,
    # ĐÃ SỬA: "llm_inspection_node" thành "web_check_node"
    ["target_llm_node", "evaluator_phase1_subgraph", "web_check_node"] 
)

# Gộp luồng: Phase 2 đợi Phase 1 và Target LLM chạy xong
eval_builder.add_edge(["target_llm_node", "evaluator_phase1_subgraph"], "evaluator_phase2_score_node")
eval_builder.add_edge("evaluator_phase2_score_node", "save_memory_node")

# Check điều kiện lặp
eval_builder.add_conditional_edges(
    "save_memory_node", 
    route_prober_loop,
    {"prober_node": "prober_node", END: END}
)
eval_builder.add_edge("prober_node", "web_check_node") # Vòng lên check lại Wiki

evaluation_subgraph = eval_builder.compile()


# ==========================================
# PHẦN 2: MAIN GRAPH (VÒNG LẶP TIẾN HÓA)
# ==========================================

class MainState(TypedDict):
    main_task: str
    categories: Dict[str, List[str]] # <-- Thêm dòng này để truyền xuống Inquirer
    taxonomy_scores: Dict[str, float]
    bad_cases_formatted: str
    current_new_task: str
    final_new_task: str
    is_terminated: bool
    seed_data: List[Dict] 
    # ĐỔI TÊN Ở ĐÂY:
    memory_pool: Annotated[List[Dict], operator.add] 

# 2. Đổi tên khi đọc ở node hút Bad Cases
def aggregate_bad_cases_node(state: MainState):
    """Hút các Bad Cases (Score <= 3.0) từ 10 luồng Evaluation để Appraiser dùng."""
    # SỬA Ở ĐÂY:
    all_records = state.get("memory_pool", [])
    bad_cases = [r for r in all_records if r.get("score", 10.0) <= 3.0]
    
    formatted = f"Found {len(bad_cases)} bad cases in recent evaluation.\n"
    for bc in bad_cases[:10]:
        formatted += f"Prompt: {bc['prompt']}\nScore: {bc['score']}\nComment: {bc['comparison']}\n\n"
        
    return {"bad_cases_formatted": formatted}

def route_appraiser_to_inquirer(state: MainState):
    """Nếu Appraiser báo hoàn hảo (is_terminated), dừng toàn bộ hệ thống."""
    if state.get("is_terminated"):
        return END
    return "inquirer_node"

def route_fan_out_evaluations(state: MainState):
    """Tuyệt chiêu Send API: Ném 10 seed cases thành 10 luồng Evaluation song song."""
    task_name = state.get("final_new_task") or "Default_Task"
    seed_cases = state.get("seed_data", [])
    
    print(f"\n[Main] Đang Fan-out {len(seed_cases)} test cases vào Evaluation Workflow...")
    
    # Tạo ra N luồng chạy song song
    return [Send("evaluation_subgraph", {
        "task_name": task_name,
        "current_case": case,
        "memory_pool": [],
        "iteration_count": 0
    }) for case in seed_cases]


# --- Build Main Graph ---
main_builder = StateGraph(MainState)

# Thêm Nodes
main_builder.add_node("appraiser_subgraph", appraiser_graph)
main_builder.add_node("inquirer_node", generate_seed_node)
main_builder.add_node("evaluation_subgraph", evaluation_subgraph)
main_builder.add_node("aggregate_bad_cases_node", aggregate_bad_cases_node)

# Cấu hình Luồng (Edges)
# main_builder.add_edge(START, "appraiser_subgraph")
main_builder.add_edge(START, "inquirer_node")

main_builder.add_conditional_edges(
    "appraiser_subgraph", 
    route_appraiser_to_inquirer,
    {"inquirer_node": "inquirer_node", END: END}
)

# Fan-out: Từ Inquirer ném ra N luồng Evaluation
main_builder.add_conditional_edges(
    "inquirer_node", 
    route_fan_out_evaluations,
    ["evaluation_subgraph"]
)

# Fan-in: Khi N luồng Evaluation xong, gom hết vào Aggregate Node
main_builder.add_edge("evaluation_subgraph", "aggregate_bad_cases_node")

# Khép kín Vòng lặp Tiến hóa (Adaptive Updating)
main_builder.add_edge("aggregate_bad_cases_node", "appraiser_subgraph")

# Compile thành Siêu Đồ Thị (Super Graph)
master_graph = main_builder.compile()