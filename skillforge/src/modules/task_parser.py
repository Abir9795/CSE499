"""
Module 1: Task Parser
Parses raw user input into structured task objects
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import re


@dataclass
class Task:
    """Structured task representation"""
    task_id: str
    description: str
    input_format: Optional[str] = None
    output_format: Optional[str] = None
    constraints: Optional[str] = None
    language: str = "python"
    difficulty_estimate: float = 0.5  # 0-1 scale
    category: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "input_format": self.input_format,
            "output_format": self.output_format,
            "constraints": self.constraints,
            "language": self.language,
            "difficulty_estimate": self.difficulty_estimate,
            "category": self.category
        }


class TaskParser:
    """Parses natural language task descriptions"""
    
    def __init__(self):
        self.task_counter = 0
        
    def parse(self, raw_input: str) -> Task:
        """
        Parse raw input into structured Task object
        
        Args:
            raw_input: Natural language task description
            
        Returns:
            Task: Structured task object
        """
        self.task_counter += 1
        
        # Extract key components from description
        description = raw_input.strip()
        
        # Basic parsing - infer input/output format
        input_format = self._extract_format(description, "input")
        output_format = self._extract_format(description, "output")
        
        # Extract constraints if mentioned
        constraints = self._extract_constraints(description)
        
        # Infer category based on keywords
        category = self._infer_category(description)
        
        return Task(
            task_id=f"task_{self.task_counter:04d}",
            description=description,
            input_format=input_format,
            output_format=output_format,
            constraints=constraints,
            category=category,
            difficulty_estimate=self._estimate_difficulty(description)
        )
    
    def _extract_format(self, text: str, ftype: str) -> Optional[str]:
        """Extract input/output format from description"""
        patterns = {
            "input": r"input:?\s*(.+?)(?:\n|$)",
            "output": r"output:?\s*(.+?)(?:\n|$)"
        }
        match = re.search(patterns[ftype], text, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def _extract_constraints(self, text: str) -> Optional[str]:
        """Extract constraints from description"""
        pattern = r"constraints?:?\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def _infer_category(self, text: str) -> str:
        """Infer task category based on keywords"""
        categories = {
            "array": ["array", "list", "vector", "matrix", "sort", "search"],
            "string": ["string", "character", "substring", "palindrome", "anagram"],
            "math": ["math", "number", "prime", "fibonacci", "gcd", "factorial"],
            "recursion": ["recursion", "recursive", "divide and conquer"],
            "oop": ["class", "object", "inheritance", "encapsulation"],
            "file": ["file", "csv", "read", "write", "json"],
            "loop": ["loop", "iteration", "for", "while", "sum"]
        }
        
        text_lower = text.lower()
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return "general"
    
    def _estimate_difficulty(self, text: str) -> float:
        """Estimate task difficulty (0-1) based on heuristics"""
        # Simple heuristic: count complexity indicators
        complexity_indicators = [
            "complex", "advanced", "optimize", "efficient", "concurrent",
            "nested", "multiple", "combination", "dynamic", "tree", "graph"
        ]
        
        text_lower = text.lower()
        indicator_count = sum(
            1 for indicator in complexity_indicators 
            if indicator in text_lower
        )
        
        # Normalize to 0-1 range
        return min(1.0, 0.3 + (indicator_count * 0.05))