import json
import re
from typing import Dict, Optional
from .utils import log_debug

def _parse_tool_call(response: str, start_tag: str, end_tag: str) -> Optional[Dict]:
    """Extracts a tool call from the model's response using dynamic boundaries."""
    start_boundary = re.escape(start_tag)
    end_boundary = re.escape(end_tag)
    pattern = rf'{start_boundary}(.*?){end_boundary}'
    
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None
    
    json_str = match.group(1).strip()
    try:
        tool_data = json.loads(json_str)
        if "arguments" in tool_data:
            tool_data["parameters"] = tool_data.pop("arguments")
        return tool_data
    except json.JSONDecodeError:
        return {"error": "malformed_json", "content": json_str}

def _compare_parameters(expected_params: Dict, actual_params: Dict) -> bool:
    """Compares expected vs. actual parameters with case-insensitive string matching."""
    if not expected_params: expected_params = {}
    if not actual_params: actual_params = {}

    if sorted(expected_params.keys()) != sorted(actual_params.keys()):
        return False

    for key, expected_value in expected_params.items():
        actual_value = actual_params[key]
        if isinstance(expected_value, str) and isinstance(actual_value, str):
            if expected_value.strip().lower() != actual_value.strip().lower():
                return False
        elif expected_value != actual_value:
            return False
    return True

def run_evaluation(config, inference_results, output_path, verbose=False):
    """Evaluates the model's responses and generates a detailed report."""
    if verbose:
        log_debug("Starting evaluation run with verbose logging enabled")
        log_debug(f"Evaluating {len(inference_results)} inference results")
    
    detailed_results = []
    
    # Get tool call boundaries from the config
    tool_call_format = config['prompt_settings']['tool_call_format']
    start_tag = tool_call_format['start_tag']
    end_tag = tool_call_format['end_tag']
    
    if verbose:
        log_debug(f"Using tool call format - start_tag: '{start_tag}', end_tag: '{end_tag}'")
    
    for i, result in enumerate(inference_results):
        expected = result["expected_behavior"]
        model_response = result["model_response"]
        parsed_call = _parse_tool_call(model_response, start_tag, end_tag)

        error_type = "CORRECT"
        if parsed_call and "error" in parsed_call:
            error_type = "MALFORMED_JSON"
        else:
            should_call = expected.get("should_call_function", False)
            if should_call:
                if not parsed_call:
                    error_type = "NO_CALL_WHEN_EXPECTED"
                elif parsed_call.get("name") != expected.get("expected_function"):
                    error_type = "WRONG_FUNCTION"
                elif not _compare_parameters(expected.get("expected_parameters"), parsed_call.get("parameters")):
                    error_type = "WRONG_PARAMETERS"
            elif parsed_call:
                error_type = "UNEXPECTED_CALL"
        
        if verbose:
            log_debug(f"EVALUATION SAMPLE {i + 1}:")
            log_debug(f"  Expected: {expected}")
            log_debug(f"  Parsed call: {parsed_call}")
            log_debug(f"  Error type: {error_type}")
            log_debug(f"  Is correct: {error_type == 'CORRECT'}")
        
        detailed_results.append({**result, "parsed_call": parsed_call, "is_correct": error_type == "CORRECT", "error_type": error_type})

    if verbose:
        log_debug(f"Evaluation completed. Processed {len(inference_results)} results")

    summary = _create_and_print_summary(detailed_results)
    
    if verbose:
        log_debug(f"Final summary: {summary}")
    
    final_report = {"summary": summary, "detailed_results": detailed_results}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

def _create_and_print_summary(results):
    """Creates, prints, and returns the evaluation summary."""
    # This logic is adapted directly from your original, robust notebook code.
    stats = {
        "total": len(results),
        "correct": sum(1 for r in results if r["is_correct"]),
        "by_scenario_type": {},
        "error_distribution": {}
    }

    for r in results:
        scenario_type = r.get("scenario_type", "unknown")
        if scenario_type not in stats["by_scenario_type"]:
            stats["by_scenario_type"][scenario_type] = {"total": 0, "correct": 0}
        stats["by_scenario_type"][scenario_type]["total"] += 1
        
        error_type = r["error_type"]
        stats["error_distribution"][error_type] = stats["error_distribution"].get(error_type, 0) + 1
        
        if r["is_correct"]:
            stats["by_scenario_type"][scenario_type]["correct"] += 1

    # Print Summary
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    print(f"🎯 Overall Accuracy: {stats['correct'] / stats['total']:.2%} ({stats['correct']}/{stats['total']})")
    print("\n📈 Accuracy by Scenario Type:")
    print("-"*80)
    for sc_type, sc_stats in sorted(stats["by_scenario_type"].items()):
        acc = (sc_stats['correct'] / sc_stats['total']) if sc_stats['total'] > 0 else 0
        print(f"{sc_type:<40} {acc:<12.2%} ({sc_stats['correct']}/{sc_stats['total']})")
    
    print("\n📉 Error Distribution:")
    print("-"*80)
    total_errors = stats['total'] - stats['correct']
    error_dist = {k: v for k, v in stats["error_distribution"].items() if k != "CORRECT"}
    for error, count in sorted(error_dist.items(), key=lambda item: item[1], reverse=True):
        perc = (count / total_errors) if total_errors > 0 else 0
        print(f"{error:<40} {count:<5} ({perc:.2%} of errors)")
    print("="*80 + "\n")
    
    return stats