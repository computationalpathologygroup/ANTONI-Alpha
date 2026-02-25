"""
VQA Answer Parser

Extracts answers from model responses for regular VQA validation.
Handles three question types: organ identification, tumor presence (yes/no), and diagnosis.
"""

import re
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List


class VQAParser:
    # Cache for taxonomy terms (loaded once)
    _taxonomy_terms = None

    @classmethod
    def _load_taxonomy_terms(cls):
        """Load all organ terms from taxonomy (cached)."""
        if cls._taxonomy_terms is not None:
            return cls._taxonomy_terms

        taxonomy_path = Path(__file__).parent.parent / 'organ' / 'taxonomy.yaml'
        with open(taxonomy_path, 'r') as f:
            taxonomy = yaml.safe_load(f)

        # Collect all terms (organ names and synonyms)
        all_terms = set()

        def collect_terms(node, name=None):
            if isinstance(node, dict):
                # Add synonyms if present
                if 'synonyms' in node:
                    for syn in node['synonyms']:
                        all_terms.add(syn.lower())

                # Add node name if provided
                if name:
                    all_terms.add(name.replace('_', ' ').lower())

                # Recurse into parts
                if 'parts' in node:
                    for part_name, part_node in node['parts'].items():
                        collect_terms(part_node, part_name)

        # Process top-level organs
        for organ_name, organ_node in taxonomy.items():
            collect_terms(organ_node, organ_name)

        cls._taxonomy_terms = all_terms
        return cls._taxonomy_terms
    """
    Parser for VQA answers.

    Handles three question types:
    - Q1 (organ): Short answer (1-4 words)
    - Q2 (tumor): Yes/No binary answer
    - Q3 (diagnosis): Diagnosis from multiple options
    """

    @staticmethod
    def parse(response: str, question_item: Dict[str, Any]) -> Optional[str]:
        """
        Parse answer from model response based on question type.

        Args:
            response: Model's generated response text
            question_item: Question dictionary containing:
                - 'question': Question text
                - 'metadata': Dict with 'question_type' key
                - 'choices': Pre-parsed choices (for diagnosis questions)

        Returns:
            Parsed answer or "unknown" if parsing fails
        """
        if not response:
            return "unknown"

        question = question_item['question']
        q_type = question_item.get('metadata', {}).get('question_type', 'unknown')

        # Determine parsing strategy based on question type
        if q_type == "organ":
            return VQAParser._parse_organ(response)
        elif q_type == "tumor":
            return VQAParser._parse_neoplasm(response)
        elif q_type == "diagnosis":
            # Use pre-parsed choices from question_item (more robust)
            options = question_item.get('choices', [])
            if not options:
                # Fallback: extract from question text if choices not provided
                options = VQAParser._extract_options_from_question(question)
            return VQAParser._parse_diagnosis(response, options)
        else:
            # Fallback: try to detect from question text
            q_lower = question.lower()
            if "which organ" in q_lower or "answer in less than four words" in q_lower:
                return VQAParser._parse_organ(response)
            elif "neoplasm" in q_lower or "answer with yes or no" in q_lower:
                return VQAParser._parse_neoplasm(response)
            elif "diagnosis" in q_lower or "likely" in q_lower or "consider:" in q_lower:
                # Use pre-parsed choices if available
                options = question_item.get('choices', [])
                if not options:
                    options = VQAParser._extract_options_from_question(question)
                return VQAParser._parse_diagnosis(response, options)

        return "unknown"

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text."""
        return re.sub(r'[^\w\s]', '', text).strip().lower()

    @staticmethod
    def _extract_options_from_question(question: str) -> List[str]:
        """Extract diagnosis options from questions like 'consider: X, Y, Z'"""
        match = re.search(r'consider:\s*([^.?]+)', question, re.IGNORECASE)
        if match:
            options_text = match.group(1)
            # Split by comma or "and"
            options = re.split(r',|\sand\s', options_text)
            return [opt.strip() for opt in options if opt.strip()]
        return []

    @classmethod
    def _parse_organ(cls, response: str) -> str:
        """
        Parse organ response using taxonomy-based matching.

        Strategy:
        1. Load all organ terms from taxonomy
        2. Search for longest matching term in response
        3. Fallback to prefix stripping if no match
        """
        response_lower = response.lower().strip()

        # Load taxonomy terms (cached)
        taxonomy_terms = cls._load_taxonomy_terms()

        # Strategy 1: Find longest matching taxonomy term
        # Sort by length (longest first) to prefer "lymph node" over "lymph"
        best_match = None
        best_match_len = 0

        for term in taxonomy_terms:
            # Check if term appears in response
            if term in response_lower:
                term_len = len(term)
                if term_len > best_match_len:
                    best_match = term
                    best_match_len = term_len

        if best_match:
            return cls._clean_text(best_match)

        # Strategy 2: Strip common verbose prefixes
        prefixes_to_remove = [
            "the image shows a histological slide of ",
            "the image shows a slide of ",
            "the image shows a section of ",
            "the image shows a ",
            "the histological slide shows a ",
            "the histological slide shows ",
            "the slide shows a ",
            "the slide shows ",
            "the residual normal tissue recognizable is ",
            "the residual normal tissue is ",
            "the residual normal tissue ",
            "this image shows a ",
            "this slide shows a ",
            "based on the provided image, the organ is ",
            "based on the image, the organ is ",
            "based on the image ",
            "the organ or residual normal tissue recognizable is ",
            "the organ is ",
        ]

        cleaned = response_lower
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break

        # Take first sentence or first few words
        first_sentence = cleaned.split('.')[0].strip()
        words = first_sentence.split()[:4]
        result = cls._clean_text(' '.join(words))

        # If result is empty or very short, fallback to original parsing
        if not result or len(result) < 3:
            first_sentence = response.split('.')[0].strip()
            words = first_sentence.split()[:4]
            result = cls._clean_text(' '.join(words))

        return result

    @staticmethod
    def _parse_neoplasm(response: str) -> str:
        """
        Parse neoplasm yes/no response. Expected format: "Yes" or "No".
        """
        response_lower = response.lower().strip()

        # Check first word/sentence for yes or no
        first_word = response_lower.split()[0] if response_lower.split() else ""

        if first_word == "yes" or response_lower.startswith("yes"):
            return "yes"
        if first_word == "no" or response_lower.startswith("no"):
            return "no"

        # Detect refusals/uncertainty
        refusal_patterns = [
            "unable to determine",
            "cannot analyze",
            "i am a text-based ai",
            "i am a language model",
            "difficult to definitively",
            "cannot definitively"
        ]
        if any(pattern in response_lower for pattern in refusal_patterns):
            return "unknown"

        # If unclear, mark as unknown
        return "unknown"

    @staticmethod
    def _parse_diagnosis(response: str, options: List[str]) -> str:
        """
        Parse diagnosis from response. Expected format: [[Diagnosis Name]].
        Falls back to robust heuristics searching from the end backwards.

        Args:
            response: Model's response text
            options: List of valid diagnosis options (pre-parsed from dataset)

        Returns:
            Extracted diagnosis (cleaned) or "unknown" if parsing fails
        """
        # Priority 1: Look for [[Diagnosis Name]] format (explicit instruction format)
        bracket_match = re.search(r'\[\[([^\]]+)\]\]', response)
        if bracket_match:
            diagnosis = bracket_match.group(1).strip()
            diagnosis_clean = VQAParser._clean_text(diagnosis)

            # Match against options if provided
            if options:
                for opt in options:
                    opt_clean = VQAParser._clean_text(opt)
                    if opt_clean == diagnosis_clean or opt_clean in diagnosis_clean or diagnosis_clean in opt_clean:
                        return opt_clean
            # If no options provided or no match, return the extracted text
            return diagnosis_clean

        # Priority 2: Detect refusals early
        response_lower = response.lower()
        refusal_patterns = [
            "further evaluation needed",
            "cannot determine",
            "unable to definitively",
            "needs more information",
            "i don't have access to"
        ]
        if any(pattern in response_lower for pattern in refusal_patterns):
            return "unknown"

        # Priority 3: Robust option matching using backward search (conclusion at end)
        if options and len(options) > 0:
            # Strategy 1: Look for strong conclusion phrases (usually at the end)
            # These explicitly signal the final answer
            conclusion_patterns = [
                # "Based on these observations, **X** is the most likely diagnosis"
                r'(?:based on|given)\s+(?:these|this|the)\s+(?:observations|findings|analysis)[,\s]+(?:the\s+)?(?:most likely|correct|best)\s+(?:diagnosis|answer)\s+(?:is|would be)\s+(?:a\s+)?(?:combination of\s+)?\*?\*?([^.*\n]+?)\*?\*?(?:[,.\n]|$)',
                # "Therefore, the final answer is: **X**"
                r'(?:therefore|thus|hence)[,\s]+(?:the\s+)?(?:final|correct)\s+(?:answer|diagnosis)\s+(?:is|would be)[:\s]*\*?\*?([^.*\n]+?)\*?\*?(?:[,.\n]|$)',
                # "The most likely diagnosis is (a combination of) **X**"
                r'(?:the\s+)?(?:most likely|final|correct|best)\s+(?:diagnosis|answer)\s+(?:is|would be)\s+(?:a\s+)?(?:combination of\s+)?\*?\*?([^.*\n]+?)\*?\*?(?:[,.\n]|$)',
                # "**X** is the most likely diagnosis"
                r'\*\*([^*]+?)\*\*\s+(?:is|are)\s+(?:the\s+)?(?:most likely|correct|best)\s+(?:diagnosis|answer)',
            ]

            for pattern in conclusion_patterns:
                matches = re.findall(pattern, response_lower, re.IGNORECASE)
                # Process matches in reverse order (prefer later matches)
                for match in reversed(matches):
                    match_clean = VQAParser._clean_text(match)
                    # Try exact match first
                    for opt in options:
                        opt_clean = VQAParser._clean_text(opt)
                        if opt_clean == match_clean:
                            return opt_clean
                    # Then try partial match
                    for opt in options:
                        opt_clean = VQAParser._clean_text(opt)
                        if opt_clean in match_clean or match_clean in opt_clean:
                            return opt_clean

            # Strategy 2: Look for bolded text (**diagnosis**) - prefer later occurrences
            bold_matches = re.findall(r'\*\*([^\*]+?)\*\*', response)

            # First, check if consecutive bold matches can be combined (e.g., "**A** and **B**")
            for i in range(len(bold_matches) - 1, 0, -1):  # Iterate backwards
                # Check if two consecutive bold segments might form a compound diagnosis
                combined = bold_matches[i-1] + " and " + bold_matches[i]
                combined_clean = VQAParser._clean_text(combined)
                for opt in options:
                    opt_clean = VQAParser._clean_text(opt)
                    if opt_clean == combined_clean:
                        return opt_clean

            # Then check individual bold matches in reverse
            for match in reversed(bold_matches):
                match_clean = VQAParser._clean_text(match)
                for opt in options:
                    opt_clean = VQAParser._clean_text(opt)
                    # Only match if reasonably similar (avoid matching single words)
                    if opt_clean == match_clean or (len(opt_clean) > 5 and opt_clean in match_clean):
                        return opt_clean

            # Strategy 3: Search backwards through the text for option mentions
            # Split into sentences and search from last to first
            sentences = re.split(r'[.!?]+', response_lower)
            for sentence in reversed(sentences):
                sentence_clean = sentence.strip()
                if not sentence_clean:
                    continue

                # Look for options in this sentence
                for opt in options:
                    opt_clean = VQAParser._clean_text(opt)
                    if opt_clean in sentence_clean:
                        # Found an option - but verify it's in a conclusive context
                        # Skip if it's in a question or listing context
                        if any(skip_word in sentence_clean for skip_word in ['consider:', 'options:', 'differential:', 'could be', 'might be', 'either']):
                            continue
                        return opt_clean

            # Strategy 4: Last resort - find last occurrence of any option in full text
            best_option = "unknown"
            last_index = -1

            for opt in options:
                opt_clean = VQAParser._clean_text(opt)
                idx = response_lower.rfind(opt_clean)
                if idx != -1 and idx > last_index:
                    last_index = idx
                    best_option = opt_clean

            if best_option != "unknown":
                return best_option

        return "unknown"


# Convenience functions for backward compatibility
def parse_organ(response: str) -> str:
    """Parse organ identification response."""
    return VQAParser._parse_organ(response)


def parse_neoplasm(response: str) -> str:
    """Parse neoplasm yes/no response."""
    return VQAParser._parse_neoplasm(response)


def parse_diagnosis(response: str, question: str) -> str:
    """Parse diagnosis response."""
    options = VQAParser._extract_options_from_question(question)
    return VQAParser._parse_diagnosis(response, options)
