
import unittest
from unittest.mock import patch

from comparison import compare_papers


class TestComparisonValidation(unittest.TestCase):

    def setUp(self):
        self.chunks = [
            {
                "paper_id": "paper-1",
                "paper_title": "Paper One",
                "text": "First paper passage",
            },
            {
                "paper_id": "paper-2",
                "paper_title": "Paper Two",
                "text": "Second paper passage",
            },
        ]

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            compare_papers(
                "",
                ["paper-1", "paper-2"],
                None,
                None,
                self.chunks,
            )

    def test_too_few_papers_rejected(self):
        with self.assertRaises(ValueError):
            compare_papers(
                "Compare methods",
                ["paper-1"],
                None,
                None,
                self.chunks,
            )

    def test_too_many_papers_rejected(self):
        with self.assertRaises(ValueError):
            compare_papers(
                "Compare methods",
                ["1", "2", "3", "4", "5", "6"],
                None,
                None,
                self.chunks,
            )

    def test_duplicate_papers_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare_papers(
                "Compare methods",
                ["paper-1", "paper-1"],
                None,
                None,
                self.chunks,
            )

    def test_invalid_top_k_rejected(self):
        with self.assertRaises(ValueError):
            compare_papers(
                "Compare methods",
                ["paper-1", "paper-2"],
                None,
                None,
                self.chunks,
                top_k_per_paper=0,
            )

    def test_missing_indexed_paper_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "missing"):
            compare_papers(
                "Compare methods",
                ["paper-1", "paper-3"],
                None,
                None,
                self.chunks,
            )

    @patch("comparison.retrieve")
    @patch("comparison.build_bm25_index")
    def test_no_retrieved_evidence_rejected(
        self,
        mock_bm25,
        mock_retrieve,
    ):
        mock_bm25.return_value = object()
        mock_retrieve.return_value = []

        with patch("comparison.bm25_retrieve", return_value=[]):
            with self.assertRaisesRegex(
                RuntimeError,
                "No relevant evidence",
            ):
                compare_papers(
                    "Compare methods",
                    ["paper-1", "paper-2"],
                    None,
                    None,
                    self.chunks,
                )

    @patch("comparison.retrieve")
    @patch("comparison.build_bm25_index")
    def test_successful_comparison_preserves_paper_coverage(
        self,
        mock_bm25,
        mock_retrieve,
    ):
        mock_bm25.return_value = object()

        mock_retrieve.side_effect = [
            [
                {
                    "paper_id": "paper-1",
                    "paper_title": "Paper One",
                    "text": "Evidence from paper one.",
                }
            ],
            [
                {
                    "paper_id": "paper-2",
                    "paper_title": "Paper Two",
                    "text": "Evidence from paper two.",
                }
            ],
        ]

        with patch(
            "comparison.bm25_retrieve",
            side_effect=[
                [],
                [],
            ],
        ):
            result = compare_papers(
                "Compare methods",
                ["paper-1", "paper-2"],
                None,
                None,
                self.chunks,
            )

        self.assertEqual(len(result["papers"]), 2)
        self.assertEqual(len(result["evidence"]), 2)

        self.assertEqual(
            {item["paper_id"] for item in result["evidence"]},
            {"paper-1", "paper-2"},
        )

        self.assertEqual(
            [item["evidence_id"] for item in result["evidence"]],
            ["E1", "E2"],
        )


if __name__ == "__main__":
    unittest.main()

