"""
VQA Dataset for Regular Validation

Implements the dataset interface for the regular validation benchmark.
Supports the 3-question format: organ, tumor presence, diagnosis.
"""

import json
import h5py
import torch
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
from huggingface_hub import hf_hub_download

from .base_dataset import BaseDataset


class VQADataset(BaseDataset):
    """
    Dataset for regular VQA validation.

    Data format:
    - Ground truth: JSON with VQA pairs (source of truth for all cases)
    - HDF5 features: For ProtoAntoni models
    - Thumbnails: For MedGemma models
    """

    REPO_ID = "SaltySander/ANTONI-Alpha-validation-data"

    def __init__(self, config: Dict):
        """
        Initialize VQA dataset.

        Args:
            config: Configuration dictionary with paths:
                - labeled_data_path: Path to ground truth JSON (source of truth)
                - h5_data_path: Path to HDF5 features
                - thumbnails_dir: Path to thumbnails directory
        """
        super().__init__(config)

        validation_config = config["validation"]
        
        # Paths
        self.labeled_data_path = Path(validation_config["labeled_data_path"])
        self.h5_data_path = Path(validation_config["h5_data_path"])
        self.thumbnails_dir = Path(validation_config["thumbnails_dir"])
        self.data_root = self.labeled_data_path.parent

        # Ensure data exists (download if needed)
        self._ensure_data_exists()

        # Load ground truth (this is now the source of truth)
        self.ground_truth = self._load_ground_truth(self.labeled_data_path)

        # Build cases from ground truth keys
        self.cases = self._build_cases_from_ground_truth()

        # Build questions list (each case has 3 questions)
        self.questions = self._build_questions()

        # HDF5 file and key map (lazy-loaded when needed by ProtoAntoni)
        self.h5_file = None
        self.h5_key_map = None

    def _ensure_data_exists(self):
        """Check if data files exist, download from HF Hub if missing."""
        self.data_root.mkdir(parents=True, exist_ok=True)

        # 1. Labeled Data
        if not self.labeled_data_path.exists():
            print(f"Downloading {self.labeled_data_path.name} from HF Hub...")
            hf_hub_download(
                repo_id=self.REPO_ID,
                filename="labeled_data_final.json",
                repo_type="dataset",
                local_dir=self.data_root,
                token=os.getenv("HF_TOKEN")
            )

        # 2. H5 Features
        if not self.h5_data_path.exists():
            print(f"Downloading {self.h5_data_path.name} from HF Hub...")
            hf_hub_download(
                repo_id=self.REPO_ID,
                filename="test_317.h5",
                repo_type="dataset",
                local_dir=self.data_root,
                token=os.getenv("HF_TOKEN")
            )

        # 3. Thumbnails
        if not self.thumbnails_dir.exists():
            tar_path = self.data_root / "thumbnails.tar.gz"
            if not tar_path.exists():
                print(f"Downloading thumbnails.tar.gz from HF Hub...")
                hf_hub_download(
                    repo_id=self.REPO_ID,
                    filename="thumbnails.tar.gz",
                    repo_type="dataset",
                    local_dir=self.data_root,
                    token=os.getenv("HF_TOKEN")
                )
            
            print(f"Extracting {tar_path.name} to {self.data_root}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=self.data_root)
            
            print(f"Extraction complete.")

    def __len__(self) -> int:
        """Return total number of questions (3 per case)."""
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single question item."""
        return self.questions[idx]

    def load_context(self, question_item: Dict[str, Any], model_type: str) -> Any:
        """
        Load context for a question.

        Args:
            question_item: Question dictionary
            model_type: "antoni_alpha" or "medgemma"

        Returns:
            - For antoni_alpha: torch.Tensor of features (num_patches, 1280)
            - For medgemma: PIL Image
        """
        context_info = question_item["context_info"]
        organ = context_info["organ"]
        case_id = context_info["case_id"]

        if model_type == "antoni_alpha":
            return self._load_h5_features(organ, case_id)
        elif model_type == "medgemma":
            return self._load_thumbnail(organ, case_id)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        # Count unique cases
        unique_cases = set()
        organs = set()
        question_types = {}

        for q in self.questions:
            context_info = q["context_info"]
            unique_cases.add((context_info["organ"], context_info["case_id"]))
            organs.add(context_info["organ"])

            q_type = q["metadata"]["question_type"]
            question_types[q_type] = question_types.get(q_type, 0) + 1

        return {
            "total_questions": len(self.questions),
            "total_cases": len(unique_cases),
            "total_organs": len(organs),
            "organs": list(organs),
            "questions_per_type": question_types,
        }

    def _build_cases_from_ground_truth(self) -> List[Dict[str, str]]:
        """Build case list from ground truth JSON keys."""
        print(f"Building cases from ground truth...")
        case_list = []

        for gt_key in self.ground_truth.keys():
            # Parse key format: histai/HISTAI-organ/case_id
            parts = gt_key.split("/")
            if len(parts) >= 3:
                organ = parts[1].lower()  # HISTAI-breast -> histai-breast
                case_id = parts[2]
                case_list.append(
                    {
                        "gt_key": gt_key,
                        "organ": organ,
                        "case_id": case_id,
                        "id": f"{organ}/{case_id}",
                    }
                )

        print(f"Built {len(case_list)} cases from ground truth.")
        return case_list

    def _load_ground_truth(self, json_path: Path) -> Dict:
        """Load ground truth JSON."""
        print(f"Loading ground truth from {json_path}...")
        with open(json_path, "r") as f:
            data = json.load(f)
        return data

    def _build_questions(self) -> List[Dict[str, Any]]:
        """Build list of questions from cases and ground truth."""
        questions = []
        question_types = ["organ", "tumor", "diagnosis"]

        for case in self.cases:
            # Get ground truth key directly from case
            gt_key = case["gt_key"]

            # Get case metadata for pre-parsed fields
            case_data = self.ground_truth[gt_key]
            vqa_pairs = case_data.get("modified_vqa", [])

            # Extract questions (user turns only)
            user_questions = []
            for i, turn in enumerate(vqa_pairs):
                if turn["role"] == "user":
                    question_text = turn["content"]
                    # Get corresponding assistant response
                    answer_text = (
                        vqa_pairs[i + 1]["content"] if i + 1 < len(vqa_pairs) else None
                    )
                    user_questions.append(
                        {"question": question_text, "answer": answer_text}
                    )

            # Create question items (expect 3 questions per case)
            for i, q_data in enumerate(user_questions):
                q_type = question_types[i] if i < len(question_types) else "unknown"

                # Modify question text for Q1 to enforce short answers
                question_text = q_data["question"]
                if q_type == "organ":
                    # Append instruction for short answer if not already present
                    if "answer in less than" not in question_text.lower():
                        question_text = f"{question_text} (Answer in less than 4 words)"

                question_item = {
                    "question_id": f"{case['id']}_q{i + 1}",
                    "question": question_text,
                    "ground_truth": q_data["answer"],
                    "context_info": {
                        "organ": case["organ"],
                        "case_id": case["case_id"],
                        "case_id_full": case["id"],
                    },
                    "metadata": {
                        "question_type": q_type,
                        "question_index": i,
                        "gt_key": gt_key,
                    },
                }

                # Add choices for perplexity evaluation
                if q_type == "tumor":
                    # Q2: Yes/No question
                    question_item["choices"] = ["Yes", "No"]
                    # Extract Yes/No from ground truth if it's a long explanation
                    question_item["ground_truth"] = self._extract_yes_no(
                        q_data["answer"]
                    )

                elif q_type == "diagnosis":
                    # Q3: Use pre-parsed differential_options and ground_truth_answer
                    # These are manually verified and should be the source of truth
                    differential_options = case_data.get("differential_options")
                    ground_truth_answer = case_data.get("ground_truth_answer")

                    if not differential_options:
                        raise ValueError(
                            f"MISSING DATA - Q3 differential_options not found for {question_item['question_id']}\n"
                            f"Case key: {gt_key}\n"
                            f"The labeled_data_final.json file should contain pre-parsed options."
                        )

                    if not ground_truth_answer:
                        raise ValueError(
                            f"MISSING DATA - Q3 ground_truth_answer not found for {question_item['question_id']}\n"
                            f"Case key: {gt_key}\n"
                            f"The labeled_data_final.json file should contain pre-extracted answer."
                        )

                    # Use pre-parsed data directly (manually verified)
                    question_item["choices"] = differential_options
                    question_item["ground_truth_full"] = q_data["answer"]
                    question_item["ground_truth"] = ground_truth_answer

                questions.append(question_item)

        return questions

    def _extract_diagnosis_choices(self, question: str) -> Optional[List[str]]:
        """
        Extract diagnosis options from Q3 question text.

        Expected formats:
        - "...consider: Option1, Option2, Option3. Which..."
        - "...consider Option1, Option2, Option3. Which..."
        """
        import re

        # Pattern 1: "consider: X, Y, Z. Which" (with colon)
        pattern1 = r"consider:\s*([^.]+)\.\s*Which"
        match = re.search(pattern1, question, re.IGNORECASE)

        if match:
            options_str = match.group(1)
            choices = self._parse_choice_list(options_str)
            return choices

        # Pattern 2: "consider X, Y, Z. Which" (without colon)
        pattern2 = r"consider\s+([^.]+)\.\s*Which"
        match = re.search(pattern2, question, re.IGNORECASE)

        if match:
            options_str = match.group(1)
            choices = self._parse_choice_list(options_str)
            return choices

        # Pattern 3: "consider: X, Y, Z. What" (with colon)
        pattern3 = r"consider:\s*([^.]+)\.\s*What"
        match = re.search(pattern3, question, re.IGNORECASE)

        if match:
            options_str = match.group(1)
            choices = self._parse_choice_list(options_str)
            return choices

        # Pattern 4: "consider X, Y, Z. What" (without colon)
        pattern4 = r"consider\s+([^.]+)\.\s*What"
        match = re.search(pattern4, question, re.IGNORECASE)

        if match:
            options_str = match.group(1)
            choices = self._parse_choice_list(options_str)
            return choices

        return None

    def _parse_choice_list(self, options_str: str) -> List[str]:
        """
        Parse a comma-separated list of choices, handling "A, B, and C" format.

        Args:
            options_str: String like "X, Y, and Z" or "X, Y, Z"

        Returns:
            List of cleaned choice strings
        """
        import re

        # Remove " and " before the last item
        # "X, Y, and Z" -> "X, Y, Z"
        options_str = re.sub(r",?\s+and\s+", ", ", options_str)

        # Split by comma and clean
        choices = []
        for opt in options_str.split(","):
            opt_clean = opt.strip()
            if opt_clean:
                choices.append(opt_clean)

        return choices

    def _extract_yes_no(self, answer: str) -> str:
        """
        Extract Yes/No from Q2 ground truth answer.

        The answer might be just "Yes" or "No", or a longer explanation.
        """
        answer_lower = answer.lower().strip()

        # Direct match
        if answer_lower in ["yes", "no"]:
            return answer.strip()

        # Check if answer starts with Yes or No
        if answer_lower.startswith("yes"):
            return "Yes"
        elif answer_lower.startswith("no"):
            return "No"

        # Fallback: return original
        return answer.strip()

    def _extract_diagnosis_from_answer(
        self, answer: str, choices: List[str]
    ) -> Optional[str]:
        """
        Extract the diagnosis name from Q3 ground truth answer.

        The answer is a long explanation with the diagnosis in bold (**diagnosis**)
        or brackets [[diagnosis]]. Match against the provided choices.
        """
        import re

        def clean_text(text: str) -> str:
            """Normalize text for comparison."""
            return re.sub(r"[^a-z0-9\s]", "", text.lower().strip())

        # Strategy 1: Look for **Diagnosis Name** (bold markdown)
        bold_matches = re.findall(r"\*\*([^\*]+)\*\*", answer)
        for match in bold_matches:
            match_clean = clean_text(match)
            for choice in choices:
                choice_clean = clean_text(choice)
                if (
                    choice_clean == match_clean
                    or choice_clean in match_clean
                    or match_clean in choice_clean
                ):
                    return choice

        # Strategy 2: Look for [[Diagnosis Name]] (brackets)
        bracket_matches = re.findall(r"\[\[([^\]]+)\]\]", answer)
        for match in bracket_matches:
            match_clean = clean_text(match)
            for choice in choices:
                choice_clean = clean_text(choice)
                if (
                    choice_clean == match_clean
                    or choice_clean in match_clean
                    or match_clean in choice_clean
                ):
                    return choice

        # Strategy 3: Direct substring match with choices
        answer_clean = clean_text(answer)
        for choice in choices:
            choice_clean = clean_text(choice)
            if choice_clean in answer_clean:
                return choice

        return None

    def _load_h5_features(self, organ: str, case_id: str) -> torch.Tensor:
        """Load HDF5 features for ProtoAntoni."""
        # Lazy-load HDF5 file and key map
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_data_path, "r")
            h5_keys = (
                list(self.h5_file["embeddings"].keys())
                if "embeddings" in self.h5_file
                else list(self.h5_file.keys())
            )
            self.h5_key_map = {}
            for k in h5_keys:
                parts = k.split("__")
                if len(parts) == 2:
                    self.h5_key_map[f"{parts[0].lower()}/{parts[1]}"] = k

        # Find key
        case_lookup = f"{organ.lower()}/{case_id}"
        h5_key = self.h5_key_map.get(case_lookup)

        if not h5_key:
            raise KeyError(f"Could not find HDF5 key for {case_lookup}")

        # Load features
        if "embeddings" in self.h5_file:
            features = self.h5_file["embeddings"][h5_key]["features"][:]
        else:
            features = self.h5_file[h5_key]["features"][:]

        return torch.from_numpy(features).float()

    def _load_thumbnail(self, organ: str, case_id: str) -> Image.Image:
        """Load thumbnail image for MedGemma."""
        # Find organ directory
        organ_dir = None
        for d in self.thumbnails_dir.iterdir():
            if d.is_dir() and d.name.lower() == organ.lower():
                organ_dir = d
                break

        if not organ_dir:
            raise RuntimeError(
                f"Organ directory not found for organ: {organ} in {self.thumbnails_dir}"
            )

        # Find image
        img_id = case_id.replace("case_", "")
        img_path = organ_dir / f"{img_id}_slide_H&E_0.jpg"

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        return Image.open(img_path).convert("RGB")

    def cleanup(self):
        """Clean up HDF5 file."""
        if self.h5_file is not None:
            try:
                self.h5_file.close()
                self.h5_file = None
            except Exception as e:
                print(f"Warning: Failed to close HDF5 file: {e}")

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass
