
import unittest

from generation import _validate_and_render


class TestCitationValidation(unittest.TestCase):
    def setUp(self):
        self.evidence = [
            {"text": "Evidence passage one."},
            {"text": "Evidence passage two."},
        ]

    def test_valid_citation_is_preserved(self):
        result = {
            "claims": [
                {
                    "text": "DFCM achieved 98% classification accuracy.",
                    "evidence_ids": ["E2"],
                }
            ],
            "insufficient_evidence": False,
        }

        answer = _validate_and_render(result, self.evidence)

        self.assertIn("[E2]", answer)
        self.assertIn("98% classification accuracy", answer)

    def test_unknown_citation_is_rejected(self):
        result = {
            "claims": [
                {
                    "text": "Unsupported claim.",
                    "evidence_ids": ["E99"],
                }
            ],
            "insufficient_evidence": False,
        }

        answer = _validate_and_render(result, self.evidence)

        self.assertIn("Insufficient evidence", answer)
        self.assertNotIn("Unsupported claim", answer)

    def test_uncited_claim_is_rejected(self):
        result = {
            "claims": [
                {
                    "text": "This claim has no citation.",
                    "evidence_ids": [],
                }
            ],
            "insufficient_evidence": False,
        }

        answer = _validate_and_render(result, self.evidence)

        self.assertIn("Insufficient evidence", answer)

    def test_duplicate_citations_are_removed(self):
        result = {
            "claims": [
                {
                    "text": "The result was reported.",
                    "evidence_ids": ["E1", "E1", "E2"],
                }
            ],
            "insufficient_evidence": False,
        }

        answer = _validate_and_render(result, self.evidence)

        self.assertEqual(answer.count("[E1]"), 1)
        self.assertEqual(answer.count("[E2]"), 1)
    def test_explicit_insufficient_evidence(self):
        result = {
            "claims": [],
            "insufficient_evidence": True,
        }

        answer = _validate_and_render(result, self.evidence)

        self.assertIn("Insufficient evidence", answer)
        self.assertIn("retrieved passages", answer)
        self.assertNotIn("citation-ID validation", answer)

    def test_invalid_claims_explain_validation_failure(self):
        result = {
            "claims": [
                {
                    "text": "GPU energy consumption was 50 Wh.",
                    "evidence_ids": ["E99"],
                }
            ],
            "insufficient_evidence": False,
        }

        answer = _validate_and_render(result, self.evidence)

        self.assertIn("Insufficient evidence", answer)
        self.assertIn("citation validation", answer)
        self.assertNotIn("50 Wh", answer)

if __name__ == "__main__":
    unittest.main()
