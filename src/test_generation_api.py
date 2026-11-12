
import json
import unittest
from unittest.mock import patch

from urllib.error import URLError

from generation import (
    _ollama_chat,
    _parse_json,
    _validate_and_render,
    generate_answer,
    generate_comparison,
)


class TestJSONParsing(unittest.TestCase):

    def test_valid_json(self):
        result = _parse_json(
            '{"claims": [], "insufficient_evidence": true}'
        )
        self.assertEqual(result["claims"], [])

    def test_markdown_fenced_json(self):
        result = _parse_json(
            '```json\n{"claims": [], "insufficient_evidence": true}\n```'
        )
        self.assertTrue(result["insufficient_evidence"])

    def test_invalid_json_rejected(self):
        with self.assertRaises(RuntimeError):
            _parse_json("not json")

    def test_non_object_rejected(self):
        with self.assertRaises(RuntimeError):
            _parse_json("[1, 2, 3]")


class TestOllamaHandling(unittest.TestCase):

    @patch("generation.urlopen")
    def test_connection_failure(self, mock_urlopen):
        mock_urlopen.side_effect = URLError("Connection refused")

        with self.assertRaisesRegex(RuntimeError, "Could not connect"):
            _ollama_chat("system", "user", 100)

    @patch("generation.urlopen")
    def test_invalid_api_json(self, mock_urlopen):
        response = mock_urlopen.return_value.__enter__.return_value
        response.read.return_value = b"not json"

        with self.assertRaisesRegex(RuntimeError, "invalid API response"):
            _ollama_chat("system", "user", 100)

    @patch("generation.urlopen")
    def test_empty_model_response(self, mock_urlopen):
        response = mock_urlopen.return_value.__enter__.return_value
        response.read.return_value = json.dumps({
            "message": {"content": ""},
            "done": True,
        }).encode()

        with self.assertRaisesRegex(RuntimeError, "empty response"):
            _ollama_chat("system", "user", 100)

    @patch("generation.urlopen")
    def test_valid_model_response(self, mock_urlopen):
        response = mock_urlopen.return_value.__enter__.return_value
        response.read.return_value = json.dumps({
            "message": {"content": "Hello"},
            "done": True,
        }).encode()

        result = _ollama_chat("system", "user", 100)

        self.assertEqual(result, "Hello")


class TestAnswerValidation(unittest.TestCase):

    def setUp(self):
        self.evidence = [
            {"text": "Evidence one"},
            {"text": "Evidence two"},
        ]

    def test_invalid_insufficient_evidence_type(self):
        result = {
            "claims": [],
            "insufficient_evidence": "false",
        }

        with self.assertRaises(RuntimeError):
            _validate_and_render(result, self.evidence)

    def test_invalid_citation_rejects_entire_claim(self):
        result = {
            "claims": [
                {
                    "text": "A factual claim.",
                    "evidence_ids": ["E1", "E99"],
                }
            ],
            "insufficient_evidence": False,
        }

        answer = _validate_and_render(result, self.evidence)

        self.assertIn("Insufficient evidence", answer)
        self.assertNotIn("A factual claim", answer)

    def test_output_word_limit(self):
        result = {
            "claims": [
                {
                    "text": " ".join(["word"] * 100),
                    "evidence_ids": ["E1"],
                },
                {
                    "text": " ".join(["extra"] * 60),
                    "evidence_ids": ["E2"],
                },
            ],
            "insufficient_evidence": False,
        }

        answer = _validate_and_render(result, self.evidence)

        claim_text = answer.split("\n\nAdditional claims")[0]
        claim_text = claim_text.rsplit(" [E1]", 1)[0]

        self.assertEqual(len(claim_text.split()), 100)
        self.assertIn("omitted", answer)
        self.assertNotIn("extra", answer)

    @patch("generation._ollama_chat")
    def test_empty_evidence_skips_model(self, mock_chat):
        answer = generate_answer("Question?", [])

        self.assertIn("Insufficient evidence", answer)
        mock_chat.assert_not_called()

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            generate_answer("", self.evidence)


class TestComparisonValidation(unittest.TestCase):

    def setUp(self):
        self.evidence = [
            {
                "evidence_id": "E1",
                "paper_id": "paper-1",
                "paper_title": "Paper One",
                "text": "First passage",
            },
            {
                "evidence_id": "E2",
                "paper_id": "paper-2",
                "paper_title": "Paper Two",
                "text": "Second passage",
            },
        ]

    @patch("generation._ollama_chat")
    def test_single_paper_comparison_rejected(self, mock_chat):
        evidence = [self.evidence[0]]

        result = generate_comparison("Compare papers", evidence)

        self.assertIn("Insufficient Evidence", result)
        mock_chat.assert_not_called()

    def test_comparison_requires_question(self):
        with self.assertRaises(ValueError):
            generate_comparison("", self.evidence)

    @patch("generation._ollama_chat")
    def test_invalid_comparison_citation_rejected(self, mock_chat):
        mock_chat.return_value = json.dumps({
            "summary": "Two papers differ.",
            "summary_evidence_ids": ["E1", "E2"],
            "findings": [
                {
                    "text": "Unsupported finding.",
                    "evidence_ids": ["E99"],
                }
            ],
            "agreements": [],
            "differences": [],
            "limitations": [],
            "insufficient_evidence": False,
        })

        result = generate_comparison("Compare papers", self.evidence)

        self.assertNotIn("Unsupported finding", result)
        self.assertIn("Two papers differ", result)

    @patch("generation._ollama_chat")
    def test_overall_summary_needs_citations_from_two_papers(self, mock_chat):
        mock_chat.return_value = json.dumps({
            "summary": "The papers have distinct approaches.",
            "summary_evidence_ids": ["E1"],
            "findings": [], "agreements": [], "differences": [],
            "limitations": [], "insufficient_evidence": False,
        })
        result = generate_comparison("Compare papers", self.evidence)
        self.assertNotIn("The papers have distinct approaches.", result)


if __name__ == "__main__":
    unittest.main()
