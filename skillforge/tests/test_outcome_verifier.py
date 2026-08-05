import unittest

from src.agents.test_generator import TestCase as GeneratedCase, TestSuite as GeneratedSuite
from src.verifier.outcome_verifier import verify


class OutcomeVerifierTests(unittest.TestCase):
    def test_verifies_list_expected_output(self):
        suite = GeneratedSuite([
            GeneratedCase(input="", expected_output=[0, 1], category="normal"),
        ])

        result = verify("print([0, 1])", suite)

        self.assertEqual(result.passed, 1)
        self.assertEqual(result.pass_rate, 1.0)

    def test_reports_structured_expected_output_on_failure(self):
        suite = GeneratedSuite([
            GeneratedCase(input="", expected_output=[0, 1], category="normal"),
        ])

        result = verify("print([1, 2])", suite)

        self.assertEqual(result.passed, 0)
        self.assertEqual(result.first_failure["expected"], "[0, 1]")
        self.assertEqual(result.failures, [result.first_failure])

    def test_limits_failure_feedback(self):
        suite = GeneratedSuite([
            GeneratedCase(input="", expected_output="1", category="normal"),
            GeneratedCase(input="", expected_output="2", category="normal"),
        ])

        result = verify("print(0)", suite, max_failures=1)

        self.assertEqual(result.passed, 0)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0]["expected"], "1")


if __name__ == "__main__":
    unittest.main()
