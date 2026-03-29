"""
State Definitions and Pydantic Schemas for Inquirer Agent

This module defines the core state objects and structured output schemas
used in the Inquirer agent's workflow. It ensures that the generated
prototype test data strictly follows the required format.
"""

from typing import Dict, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class InquirerState(TypedDict):
    """
    Internal State for the Inquirer sub-graph.
    """
    # Input
    task_name: str
    categories: Dict[str, List[str]] # The full taxonomy for context
    
    # Internal & Output
    retry_count: int
    seed_data: List[Dict] # The final list of 10 test cases


class PromptContent(BaseModel):
    """Schema for the actual prompt content (claim and auxiliary info)."""
    source_claim: str = Field(
        description="The statement or claim to be fact-checked."
    )
    auxiliary_info: str = Field(
        description="External knowledge source. Must be empty if test_mode is [claim]. Otherwise, contains Wikipedia evidence or social media conversation thread."
    )

class TestCase(BaseModel):
    """Schema for a single fact-checking test case."""
    key_point: str = Field(
        description="A short sentence summarizing the key point to test the LLM."
    )
    test_mode: Literal["[claim]", "[evidence]", "[wisdom of crowds]"] = Field(
        description="The problem setting of the fact-checking task."
    )
    prompt: PromptContent = Field(
        description="The core content of the test case."
    )

class InquirerOutput(BaseModel):
    """
    Schema for the final output of the Inquirer LLM.
    Forces the LLM to return exactly a list of Test Cases.
    """
    test_cases: List[TestCase] = Field(
        description="A list containing exactly 10 generated test cases.",
        min_items=10, # Bắt buộc LLM phải đẻ đủ 10 cases
        max_items=10
    )