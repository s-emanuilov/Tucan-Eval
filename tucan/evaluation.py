#!/usr/bin/env python3
"""
Evaluation module for the Tucan framework.
"""

import json
from typing import Dict, List, Any, Optional
from .evaluate import _parse_tool_call, _compare_parameters


class FunctionCallEvaluator:
    """Evaluator for function calling tasks."""

    def __init__(self):
        """Initialize the function call evaluator."""
        pass

    def evaluate_results(
        self, inference_results: List[Dict[str, Any]], tool_call_format: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Evaluate inference results for function calling accuracy.

        Args:
            inference_results: List of inference result dictionaries
            tool_call_format: Tool call format configuration

        Returns:
            Dictionary containing evaluation results and summary
        """
        detailed_results = []

        start_tag = tool_call_format["start_tag"]
        end_tag = tool_call_format["end_tag"]

        for i, result in enumerate(inference_results):
            expected = result["expected_behavior"]
            model_response = result["model_response"]

            # Skip results with errors
            if result.get("error", False):
                detailed_results.append(
                    {
                        **result,
                        "parsed_call": None,
                        "is_correct": False,
                        "error_type": "API_ERROR",
                    }
                )
                continue

            # Parse tool call from response
            parsed_call = self._parse_tool_call(model_response, start_tag, end_tag)

            # Determine correctness
            error_type = self._determine_error_type(expected, parsed_call)

            detailed_results.append(
                {
                    **result,
                    "parsed_call": parsed_call,
                    "is_correct": error_type == "CORRECT",
                    "error_type": error_type,
                }
            )

        # Create summary
        summary = self._create_summary(detailed_results)

        return {"detailed_results": detailed_results, "summary": summary}

    def _parse_tool_call(
        self, response: str, start_tag: str, end_tag: str
    ) -> Optional[Dict]:
        """Parse a tool call from the model's response."""
        return _parse_tool_call(response, start_tag, end_tag)

    def _determine_error_type(self, expected: Any, parsed_call: Optional[Dict]) -> str:
        """Determine the type of error (if any) in the model's response."""
        if parsed_call and "error" in parsed_call:
            return "MALFORMED_JSON"

        # Handle case where expected_behavior might be a string instead of dict
        if isinstance(expected, str):
            try:
                # Try to parse it as JSON in case it's a JSON string
                expected = json.loads(expected)
            except (json.JSONDecodeError, ValueError):
                # If it can't be parsed as JSON, it's a data format error
                return "DATA_FORMAT_ERROR"
        
        # Expected should be a dictionary
        if not isinstance(expected, dict):
            return "DATA_FORMAT_ERROR"

        should_call = expected.get("should_call_function", False)

        if should_call:
            if not parsed_call:
                return "NO_CALL_WHEN_EXPECTED"
            elif parsed_call.get("name") != expected.get("expected_function"):
                return "WRONG_FUNCTION"
            elif not _compare_parameters(
                expected.get("expected_parameters"), parsed_call.get("parameters")
            ):
                return "WRONG_PARAMETERS"
        elif parsed_call:
            return "UNEXPECTED_CALL"

        return "CORRECT"

    def _create_summary(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """Create evaluation summary statistics."""
        total = len(detailed_results)
        correct = sum(1 for r in detailed_results if r["is_correct"])

        # Group by scenario type
        by_scenario_type = {}
        error_distribution = {}

        for result in detailed_results:
            scenario_type = result.get("scenario_type", "unknown")
            error_type = result["error_type"]

            # Track by scenario type
            if scenario_type not in by_scenario_type:
                by_scenario_type[scenario_type] = {"total": 0, "correct": 0}
            by_scenario_type[scenario_type]["total"] += 1
            if result["is_correct"]:
                by_scenario_type[scenario_type]["correct"] += 1

            # Track error distribution
            error_distribution[error_type] = error_distribution.get(error_type, 0) + 1

        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "by_scenario_type": by_scenario_type,
            "error_distribution": error_distribution,
        }
