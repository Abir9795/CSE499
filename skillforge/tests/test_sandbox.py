import unittest

from src.verifier.sandbox import run_code, Status


class SandboxTests(unittest.TestCase):
    def test_run_code_accepts_list_stdin_input(self):
        result = run_code("import sys\nprint(sys.stdin.read())", stdin_input=[1, 2, 3])
        self.assertEqual(result.status, Status.PASS)
        self.assertEqual(result.stdout.strip(), "1\n2\n3")


if __name__ == "__main__":
    unittest.main()
