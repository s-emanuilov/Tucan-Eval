import json
import os
from typing import Dict, List, Any, Optional
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download, list_repo_files
import logging

logger = logging.getLogger(__name__)


def _parse_functions_from_dataset(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse function strings from HuggingFace dataset samples into proper JSON objects.
    
    Args:
        samples: List of samples from the dataset
        
    Returns:
        List of samples with parsed function objects
    """
    processed_samples = []
    functions_converted = 0
    
    for sample in samples:
        processed_sample = sample.copy()
        
        # Check if functions field exists and is a string
        if 'functions' in processed_sample and isinstance(processed_sample['functions'], str):
            try:
                # Parse the JSON string into proper objects
                parsed_functions = json.loads(processed_sample['functions'])
                processed_sample['functions'] = parsed_functions
                functions_converted += 1
                logger.debug(f"Successfully parsed functions string to JSON objects")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse functions JSON string: {e}")
                # Keep the original string if parsing fails
                pass
        
        processed_samples.append(processed_sample)
    
    if functions_converted > 0:
        print(f"🔄 Converted {functions_converted} function strings to JSON objects")
    
    return processed_samples


def load_samples_from_source(
    source: str,
    source_type: str = "auto",
    split: Optional[str] = None,
    subset: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Load evaluation samples from various sources.

    Args:
        source: The source to load from. Can be:
            - Local file path (JSON)
            - HuggingFace dataset name (e.g., "username/dataset-name")
            - HuggingFace file path (e.g., "username/repo-name/file.json")
        source_type: Type of source ("auto", "local", "hf_dataset", "hf_file")
        split: Dataset split to load (for HF datasets, e.g., "train", "test")
        subset: Dataset subset/configuration (for HF datasets)
        token: HuggingFace token for private/gated content
        **kwargs: Additional arguments passed to dataset loading functions

    Returns:
        List of sample dictionaries

    Raises:
        ValueError: If the source type cannot be determined or is invalid
        FileNotFoundError: If local file is not found
        Exception: For HuggingFace API errors
    """

    # Auto-detect source type if not specified
    if source_type == "auto":
        source_type = _detect_source_type(source)

    print(f"📥 Loading samples from {source_type}: {source}")

    if source_type == "local":
        return _load_local_json(source)
    elif source_type == "hf_dataset":
        return _load_hf_dataset(
            source, split=split, subset=subset, token=token, **kwargs
        )
    elif source_type == "hf_file":
        return _load_hf_file(source, token=token)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


def _detect_source_type(source: str) -> str:
    """Auto-detect the type of source based on the path/name."""

    # Check if it's a local file
    if os.path.exists(source) and os.path.isfile(source):
        return "local"

    # Check if it contains a file extension (likely HF file)
    if "/" in source and any(
        source.endswith(ext) for ext in [".json", ".jsonl", ".csv", ".parquet"]
    ):
        return "hf_file"

    # Default to HF dataset if it contains "/"
    if "/" in source:
        return "hf_dataset"

    # If no "/" and not a local file, assume it's a local file that doesn't exist yet
    return "local"


def _load_local_json(file_path: str) -> List[Dict[str, Any]]:
    """Load samples from a local JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        if not isinstance(samples, list):
            raise ValueError("JSON file must contain a list of samples")

        print(f"✅ Loaded {len(samples)} samples from local file")
        return samples

    except FileNotFoundError:
        raise FileNotFoundError(f"Local file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON file {file_path}: {e}")


def _load_hf_dataset(
    dataset_name: str,
    split: Optional[str] = None,
    subset: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load samples from a HuggingFace dataset."""
    try:
        # Load the dataset
        dataset = load_dataset(
            dataset_name, name=subset, split=split, token=token, **kwargs
        )

        # Convert to list of dictionaries
        if isinstance(dataset, Dataset):
            samples = dataset.to_list()
        else:
            # Handle DatasetDict case
            if split is None:
                # If no split specified, try common split names
                available_splits = list(dataset.keys())
                if "test" in available_splits:
                    split = "test"
                elif "validation" in available_splits:
                    split = "validation"
                elif "train" in available_splits:
                    split = "train"
                else:
                    split = available_splits[0]

                print(
                    f"ℹ️  No split specified, using '{split}' (available: {available_splits})"
                )

            samples = dataset[split].to_list()

        # Parse function strings into JSON objects if needed
        samples = _parse_functions_from_dataset(samples)

        print(f"✅ Loaded {len(samples)} samples from HuggingFace dataset")
        return samples

    except Exception as e:
        raise Exception(f"Error loading HuggingFace dataset {dataset_name}: {e}")


def _load_hf_file(file_path: str, token: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load samples from a specific file in a HuggingFace repository."""
    try:
        # Parse the file path (format: username/repo-name/path/to/file.json)
        parts = file_path.split("/")
        if len(parts) < 3:
            raise ValueError(
                "HuggingFace file path must be in format: username/repo-name/file.json"
            )

        repo_id = "/".join(parts[:2])  # username/repo-name
        filename = "/".join(parts[2:])  # path/to/file.json

        print(f"📥 Downloading {filename} from {repo_id}")

        # Download the file
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)

        # Load the downloaded file
        with open(local_path, "r", encoding="utf-8") as f:
            if filename.endswith(".json"):
                samples = json.load(f)
            elif filename.endswith(".jsonl"):
                samples = [json.loads(line) for line in f if line.strip()]
            else:
                raise ValueError(f"Unsupported file format: {filename}")

        if not isinstance(samples, list):
            raise ValueError("File must contain a list of samples")

        # Parse function strings into JSON objects if needed
        samples = _parse_functions_from_dataset(samples)

        print(f"✅ Loaded {len(samples)} samples from HuggingFace file")
        return samples

    except Exception as e:
        raise Exception(f"Error loading HuggingFace file {file_path}: {e}")


def list_available_files(repo_id: str, token: Optional[str] = None) -> List[str]:
    """
    List available files in a HuggingFace repository.

    Args:
        repo_id: Repository ID (e.g., "username/repo-name")
        token: HuggingFace token for private repositories

    Returns:
        List of file paths in the repository
    """
    try:
        files = list_repo_files(repo_id, token=token)
        # Filter for common data files
        data_files = [
            f
            for f in files
            if any(f.endswith(ext) for ext in [".json", ".jsonl", ".csv", ".parquet"])
        ]
        return data_files
    except Exception as e:
        raise Exception(f"Error listing files in repository {repo_id}: {e}")


def preview_dataset_info(
    source: str, source_type: str = "auto", token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get preview information about a dataset without loading all samples.

    Args:
        source: Dataset source
        source_type: Type of source
        token: HuggingFace token

    Returns:
        Dictionary with dataset information
    """

    if source_type == "auto":
        source_type = _detect_source_type(source)

    info = {"source": source, "type": source_type}

    try:
        if source_type == "hf_dataset":
            # Load just the first few samples to get structure
            dataset = load_dataset(source, split="train[:5]", token=token)
            samples = dataset.to_list()
            info.update(
                {
                    "sample_count": "5+ (preview only)",
                    "sample_structure": list(samples[0].keys()) if samples else [],
                    "preview_samples": samples[:2],
                }
            )

        elif source_type == "local" and os.path.exists(source):
            with open(source, "r", encoding="utf-8") as f:
                samples = json.load(f)
            info.update(
                {
                    "sample_count": len(samples),
                    "sample_structure": list(samples[0].keys()) if samples else [],
                    "preview_samples": samples[:2],
                }
            )

        elif source_type == "hf_file":
            # For HF files, we need to download and peek
            parts = source.split("/")
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])

            available_files = list_available_files(repo_id, token)
            info.update({"available_files": available_files, "target_file": filename})

    except Exception as e:
        info["error"] = str(e)

    return info
