"""
Module 2: Code Generator
Generates code using local LLM
"""
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import json
import logging

# Import with error handling
try:
    from langchain_ollama import OllamaLLM
except ModuleNotFoundError as exc:
    if exc.name != "langchain_ollama":
        raise
    OllamaLLM = None  # type: ignore
    _OLLAMA_IMPORT_ERROR = exc
else:
    _OLLAMA_IMPORT_ERROR = None

try:
    from langchain_core.prompts import PromptTemplate
except ModuleNotFoundError as exc:
    if exc.name != "langchain_core":
        raise
    PromptTemplate = None  # type: ignore
    _LANGCHAIN_CORE_ERROR = exc
else:
    _LANGCHAIN_CORE_ERROR = None

try:
    from langchain_core.output_parsers import StrOutputParser
except ModuleNotFoundError as exc:
    if exc.name != "langchain_core":
        raise
    StrOutputParser = None  # type: ignore
else:
    pass

try:
    from src.modules.task_parser import Task
except ModuleNotFoundError as exc:
    if exc.name != "src.modules.task_parser":
        raise
    Task = None  # type: ignore
    _TASK_PARSER_ERROR = exc
else:
    _TASK_PARSER_ERROR = None

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Generates Python code from task descriptions"""
    
    def __init__(
        self, 
        model_name: str = "qwen2.5-coder:7b",
        temperature: float = 0.3,
        max_retries: int = 3
    ):
        # Check all required imports
        if OllamaLLM is None:
            raise ImportError(
                "Missing dependency 'langchain-ollama'. Install with: "
                "pip install -r requirements.txt"
            ) from _OLLAMA_IMPORT_ERROR
        
        if PromptTemplate is None or StrOutputParser is None:
            raise ImportError(
                "Missing dependency 'langchain-core'. Install with: "
                "pip install -r requirements.txt"
            ) from _LANGCHAIN_CORE_ERROR
        
        if Task is None:
            raise ImportError(
                "Missing module 'src.modules.task_parser'. Ensure the module exists."
            ) from _TASK_PARSER_ERROR
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Initialize Ollama LLM
        self.llm = OllamaLLM(
            model=model_name,
            temperature=temperature,
            num_predict=4096,
            stop=["```", "\n\n\n"]
        )
        
        # Define code generation prompt
        self.code_prompt = PromptTemplate(
            input_variables=["task", "skills", "language"],
            template="""You are an expert Python programmer. Write clean, efficient, and correct code.

Task: {task}

{skills}

Language: {language}

Requirements:
1. Write only the function/class, no explanation text
2. Include proper error handling
3. Use type hints where appropriate
4. Follow PEP 8 style
5. Ensure the code is self-contained (no external imports unless necessary)

Return ONLY the code in a markdown code block:

```python
# Generated code:"""
        )
        
        self.parser = StrOutputParser()
    
    def generate_code(
        self, 
        task: Task,
        skills: str = "",
        language: str = "Python"
    ) -> Dict[str, Any]:
        """
        Generate code from a task description
        
        Args:
            task: Task object with description and requirements
            skills: Available skills context for code generation
            language: Programming language (default: Python)
            
        Returns:
            Dictionary with generated code and metadata
        """
        try:
            # Build the prompt
            prompt_text = self.code_prompt.format(
                task=task.description,
                skills=skills or "No specific skills provided",
                language=language
            )
            
            # Generate code with retries
            code = None
            for attempt in range(self.max_retries):
                try:
                    code = self.llm.invoke(prompt_text)
                    if code and len(code.strip()) > 0:
                        break
                    logger.warning(f"Attempt {attempt + 1}: Empty response from LLM")
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        raise
            
            if not code:
                raise ValueError("Failed to generate code after all retries")
            
            # Clean up the generated code
            code = self._clean_code(code)
            
            return {
                "code": code,
                "task_id": task.task_id,
                "model": self.model_name,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {
                "code": "",
                "task_id": task.task_id,
                "model": self.model_name,
                "status": "error",
                "error": str(e)
            }
    
    def _clean_code(self, code: str) -> str:
        """
        Clean generated code by removing markdown formatting
        
        Args:
            code: Raw generated code
            
        Returns:
            Cleaned code
        """
        # Remove markdown code blocks
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()
    
    def generate_multiple(
        self,
        tasks: List[Task],
        skills: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Generate code for multiple tasks
        
        Args:
            tasks: List of Task objects
            skills: Available skills context for all tasks
            
        Returns:
            List of generated code results
        """
        results = []
        for task in tasks:
            result = self.generate_code(task, skills)
            results.append(result)
            logger.info(
                f"Generated code for task {task.task_id}: {result['status']}"
            )
        
        return results
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate generated Python code
        
        Args:
            code: Python code to validate
            
        Returns:
            Validation results
        """
        try:
            compile(code, '<string>', 'exec')
            return {
                "valid": True,
                "error": None
            }
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            return {
                "valid": False,
                "error": str(e),
                "line": e.lineno
            }
        except Exception as e:
            logger.error(f"Code validation error: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
        
