"""Language identifiers shared by datasets and execution backends."""

from enum import Enum


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    CPP = "cpp"
    JAVA = "java"

    def __str__(self):
        return self.value
