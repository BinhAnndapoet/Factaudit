from typing import Dict, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class InspectorState(TypedDict):
    task_name: str
    pending_cases: List[Dict]      # Danh sách test case chờ duyệt (Tới từ Inquirer hoặc Prober)
    approved_cases: List[Dict]     # Danh sách test case đã đạt chuẩn
    current_case: Optional[Dict]   # Test case đang được xử lý ở vòng lặp hiện tại
    retry_count: int               # Đếm số lần sửa chữa cho current_case


class PromptContent(BaseModel):
    source_claim: str = Field(description="The statement to be fact-checked.")
    auxiliary_info: str = Field(description="External knowledge source or empty.")

class TestCase(BaseModel):
    key_point: str = Field(description="Short sentence summarizing the key point.")
    test_mode: Literal["[claim]", "[evidence]", "[wisdom of crowds]"] = Field(description="Problem setting.")
    prompt: PromptContent = Field(description="Core content.")

class InspectionOutput(BaseModel):
    is_valid: bool = Field(description="True if the test case meets all criteria, False otherwise.")
    feedback: str = Field(default="", description="If invalid, explain what rules were violated.")
    revised_case: TestCase = Field(description="The valid test case (either original if valid, or a corrected version).")