import unittest

from src.verifier.sandbox import run_code, run_code_many, Status


class SandboxTests(unittest.TestCase):
    def test_run_code_accepts_list_stdin_input(self):
        result = run_code("import sys\nprint(sys.stdin.read())", stdin_input=[1, 2, 3])
        self.assertEqual(result.status, Status.PASS)
        self.assertEqual(result.stdout.strip(), "1\n2\n3")

    def test_run_code_many_reuses_candidate_file(self):
        results = run_code_many(
            "value = int(input())\nprint(value * 2)",
            ["2", "4"],
        )

        self.assertEqual([result.status for result in results], [Status.PASS, Status.PASS])
        self.assertEqual([result.stdout.strip() for result in results], ["4", "8"])


if __name__ == "__main__":
    unittest.main()
