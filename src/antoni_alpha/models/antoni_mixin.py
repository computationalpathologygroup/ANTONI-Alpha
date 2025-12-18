"""
Shared business logic for ANTONI-Alpha LLaVA-style multimodal models.

This mixin contains all the core functionality shared between the original
nn.Module-based implementation and the HuggingFace PreTrainedModel version.
"""

import torch
import logging

logger = logging.getLogger(__name__)


class AntoniAlphaMixin:
    """
    Mixin class containing shared business logic for LLaVA multimodal models.

    This class should be used in multiple inheritance alongside either nn.Module
    or PreTrainedModel. It contains all the core functionality for:
    - Component freezing/unfreezing
    - Label creation and masking
    - Vision embedding insertion
    - Forward pass logic
    - Text generation
    - Multiple-choice evaluation

    Expected attributes (set by concrete classes in __init__):
        - self.config: AntoniAlphaConfig instance
        - self.llm: PEFT-wrapped LLM model
        - self.projection_layer: AntoniProjector instance
        - self.processor: AutoProcessor instance
        - self.llm_hidden_size: int
    """

    def freeze_component(self, component_name: str, freeze: bool = True):
        """
        Freeze or unfreeze parameters of a model component.

        Args:
            component_name: Component to modify ('projection' or 'llm')
            freeze: If True, freeze parameters; if False, unfreeze them
        """
        component_map = {
            "projection": self.projection_layer,
            "llm": self.llm,
        }
        if component_name not in component_map:
            raise ValueError(
                f"Unknown component '{component_name}'. Options are: {list(component_map.keys())}"
            )

        for param in component_map[component_name].parameters():
            param.requires_grad = not freeze

        logger.info(
            f"Component '{component_name}' has been {'frozen' if freeze else 'unfrozen'}."
        )

    def create_assistant_only_labels(
        self, input_ids: torch.Tensor, final_text: list[str]
    ) -> torch.Tensor:
        """
        Create labels that only compute loss on assistant responses.

        Masks all content except text between <start_of_turn>model and <end_of_turn> tags.

        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len]
            final_text: Final text strings that were tokenized

        Returns:
            Labels with user portions masked as -100
        """
        labels = input_ids.clone()

        # For each sample in the batch
        for batch_idx, text in enumerate(final_text):
            # Find all assistant response regions in the text
            current_labels = labels[batch_idx].clone()

            # Tokenize the text to get positions
            tokenized = self.processor.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=False,
                return_offsets_mapping=True,
            )

            # Get token IDs for matching
            token_ids = tokenized.input_ids[0]

            # Mask everything initially, then unmask only assistant parts
            current_labels[:] = -100

            # Find assistant response regions using simple string matching
            model_start_tag = "<start_of_turn>model"
            model_end_tag = "<end_of_turn>"

            start_idx = 0
            while True:
                # Find the next model response
                model_start = text.find(model_start_tag, start_idx)
                if model_start == -1:
                    break

                model_end = text.find(model_end_tag, model_start)
                if model_end == -1:
                    break

                # Include tags in training for proper generation
                full_response_text = text[model_start:model_end + len(model_end_tag)]

                # Avoid adding extra <bos> token
                response_tokens = self.processor.tokenizer(
                    full_response_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                ).input_ids[0]

                # Find where this response appears in the full tokenization
                for i in range(len(token_ids) - len(response_tokens) + 1):
                    if torch.equal(
                        token_ids[i : i + len(response_tokens)], response_tokens
                    ):
                        # Unmask this region
                        seq_len = current_labels.shape[0]
                        end_pos = min(i + len(response_tokens), seq_len)
                        if end_pos <= seq_len:
                            current_labels[i:end_pos] = token_ids[i:end_pos]
                        break

                start_idx = model_end + len(model_end_tag)

            labels[batch_idx] = current_labels

        # Mask special tokens
        image_token_id = getattr(
            self.processor.tokenizer,
            "image_token_id",
            self.processor.tokenizer.convert_tokens_to_ids("<image>"),
        )
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100

        return labels

    def add_image_tokens_to_text(self, text_input: list[str]) -> list[str]:
        """
        Adds the full image placeholder sequence to the beginning of text inputs for inference.

        Args:
            text_input: List of text strings

        Returns:
            Modified text with image token added
        """
        image_placeholder_string = self.processor.full_image_sequence
        return [image_placeholder_string + text for text in text_input]

    def insert_vision_embeddings(
        self,
        text_embeddings: torch.Tensor,
        projected_vision_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Insert vision embeddings at image token positions using masked_scatter.

        Args:
            text_embeddings: Text embeddings including image token positions [batch_size, seq_len, hidden_dim]
            projected_vision_embeddings: Projected vision features [batch_size, num_vision_tokens, hidden_dim]
            input_ids: Token IDs to identify image token positions [batch_size, seq_len]

        Returns:
            Tuple of (modified_embeddings, attention_mask)
        """
        # Use the processor's image token ID for consistency
        image_token_id = getattr(
            self.processor.tokenizer,
            "image_token_id",
            self.processor.tokenizer.convert_tokens_to_ids("<image>"),
        )

        # Create mask for image token positions
        image_token_mask = input_ids == image_token_id  # [batch_size, seq_len]

        # Expand mask to embedding dimension
        image_token_mask_expanded = image_token_mask.unsqueeze(-1).expand_as(
            text_embeddings
        )
        # [batch_size, seq_len, hidden_dim]

        # Verify dimensions match (like Gemma3 does)
        n_image_tokens = image_token_mask.sum()
        n_image_features = (
            projected_vision_embeddings.shape[0] * projected_vision_embeddings.shape[1]
        )

        if (
            text_embeddings[image_token_mask_expanded].numel()
            != projected_vision_embeddings.numel()
        ):
            raise ValueError(
                f"Image features and image tokens do not match: "
                f"tokens: {n_image_tokens}, features {n_image_features}"
            )

        # Replace masked positions with vision embeddings
        modified_embeddings = text_embeddings.masked_scatter(
            image_token_mask_expanded,
            projected_vision_embeddings.reshape(
                -1, projected_vision_embeddings.shape[-1]
            ),
        )

        # Create attention mask (1 for all valid tokens, 0 for padding)
        attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long()

        return modified_embeddings, attention_mask

    def prepare_labels_with_image_masking(
        self, labels: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Mask image tokens in labels (they shouldn't contribute to loss).

        Args:
            labels: Label tensor (already has user portions masked if in training)
            input_ids: Token IDs to identify image token positions

        Returns:
            Labels with image tokens masked
        """
        image_token_id = getattr(
            self.processor.tokenizer,
            "image_token_id",
            self.processor.tokenizer.convert_tokens_to_ids("<image>"),
        )

        # Mask image tokens
        modified_labels = labels.clone()
        modified_labels[input_ids == image_token_id] = -100

        # Also mask padding tokens if not already done
        modified_labels[input_ids == self.processor.tokenizer.pad_token_id] = -100

        return modified_labels

    def forward(
        self,
        slide_latents: torch.Tensor,
        conversations: list = None,
        text_input: list[str] = None,
    ):
        """
        Forward pass with vision embedding insertion.

        Args:
            slide_latents: Input slide latents [batch_size, num_latents, embedding_dim]
            conversations: Conversation lists in OpenAI format (training mode)
            text_input: Input text strings (inference mode)

        Returns:
            CausalLMOutputWithPast from the LLM
        """
        llm_device = next(self.llm.parameters()).device

        # Ensure input tensor has correct dtype
        slide_latents = slide_latents.to(
            dtype=next(self.projection_layer.parameters()).dtype
        )

        # 1. Project vision embeddings to the LLM's input space
        projected_vision_embeddings = self.projection_layer(slide_latents)

        # 2. Handle text input with image tokens
        if conversations is not None:
            # Training path: Implement the exact two-stage templating and expansion

            # Stage 1: Modify conversation structure for the templater
            # We assume a batch of conversations, and we modify the first user turn in each.
            processed_conversations = []
            for conv in conversations:
                modified_conv = []
                found_first_user = False
                for turn in conv:
                    if not found_first_user and turn["role"] == "user":
                        # Reformat content from a string to a list of dicts
                        original_content = turn["content"]
                        modified_content = [
                            {"type": "image"},
                            {"type": "text", "text": original_content},
                        ]
                        modified_turn = {"role": "user", "content": modified_content}
                        modified_conv.append(modified_turn)
                        found_first_user = True
                    else:
                        modified_conv.append(turn)
                processed_conversations.append(modified_conv)

            # Stage 2: Apply template to get string with simple placeholder, then expand it.
            # a) Apply template to create the basic text structure
            templated_text = self.processor.tokenizer.apply_chat_template(
                processed_conversations,
                tokenize=False,
                add_generation_prompt=False,
            )

            # b) Expand the simple image placeholder with the full sequence
            # The chat template produces <start_of_image> but we need to replace it with the full sequence
            final_text = [
                s.replace("<start_of_image>", self.processor.full_image_sequence)
                for s in templated_text
            ]

            # c) Tokenize the final, complete string
            text_inputs = self.processor.tokenizer(
                final_text,
                return_tensors="pt",
                padding=True,
            ).to(llm_device)

            # Create labels with assistant-only masking
            labels = self.create_assistant_only_labels(
                text_inputs.input_ids, final_text
            )
            input_ids = text_inputs.input_ids

        elif text_input is not None:
            # Inference path: add image tokens to text
            modified_text = self.add_image_tokens_to_text(text_input)

            inputs = self.processor(
                text=modified_text, return_tensors="pt", padding=True
            ).to(llm_device)
            input_ids = inputs.input_ids
            labels = input_ids.clone()
        else:
            raise ValueError("Either conversations or text_input must be provided")

        # 3. Get text embeddings from the LLM
        text_embeddings = self.llm.get_input_embeddings()(input_ids)

        # 4. Insert vision embeddings at image token positions
        inputs_embeds, attention_mask = self.insert_vision_embeddings(
            text_embeddings,
            projected_vision_embeddings.to(
                device=text_embeddings.device, dtype=text_embeddings.dtype
            ),
            input_ids,
        )

        # 5. Prepare labels with image token masking
        final_labels = self.prepare_labels_with_image_masking(labels, input_ids)

        # 6. Perform the forward pass through the LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=final_labels,
            return_dict=True,
        )

        return outputs

    def generate(
        self,
        slide_latents: torch.Tensor,
        conversations: list[list[dict]],
        **generation_kwargs,
    ):
        """
        Generate text responses using the multimodal model.

        Args:
            slide_latents (torch.Tensor): Input tensor of slide latents.
                                        Shape: [batch_size, num_latents, embedding_dim]
            conversations (list[list[dict]]): A batch of conversation lists in OpenAI format.
                                            Each conversation is a list of turns with "role" and "content".
                                            Example single-turn: [[{"role": "user", "content": "Describe this"}]]
                                            Example multi-turn: [[{"role": "user", "content": "..."},
                                                                  {"role": "assistant", "content": "..."},
                                                                  {"role": "user", "content": "..."}]]
            **generation_kwargs: Additional arguments passed to the LLM's generate method.

        Returns:
            torch.Tensor: Generated token IDs from the LLM.
        """
        llm_device = next(self.llm.parameters()).device

        # Ensure input tensor has correct dtype
        slide_latents = slide_latents.to(
            dtype=next(self.projection_layer.parameters()).dtype
        )

        # 1. Project vision embeddings to the LLM's input space
        projected_vision_embeddings = self.projection_layer(slide_latents)

        # 2. Prepare conversations with proper image token placement
        # Ensure first user turn has image token (auto-inject if missing)
        processed_conversations = []
        for conv in conversations:
            processed_conv = []
            first_user_found = False

            for turn in conv:
                if not first_user_found and turn["role"] == "user":
                    # First user turn - ensure it has image token
                    content = turn["content"]

                    # Check if content is already in multimodal format
                    if isinstance(content, list):
                        # Already structured - check if image token exists
                        has_image = any(item.get("type") == "image" for item in content)
                        if has_image:
                            processed_conv.append(turn)
                        else:
                            # Add image token at the beginning
                            processed_conv.append({
                                "role": "user",
                                "content": [{"type": "image"}] + content,
                            })
                    else:
                        # Plain string - convert to multimodal format with image
                        processed_conv.append({
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": content},
                            ],
                        })
                    first_user_found = True
                else:
                    # All other turns - pass through as-is
                    processed_conv.append(turn)

            processed_conversations.append(processed_conv)

        templated_text = self.processor.tokenizer.apply_chat_template(
            processed_conversations,
            tokenize=False,
            add_generation_prompt=True,
        )

        final_text = [
            s.replace("<start_of_image>", self.processor.full_image_sequence)
            for s in templated_text
        ]

        # 3. Tokenize the final, complete string
        inputs = self.processor.tokenizer(
            final_text,
            return_tensors="pt",
            padding=True,
        ).to(llm_device)
        input_ids = inputs.input_ids

        # 4. Get text embeddings from the LLM
        text_embeddings = self.llm.get_input_embeddings()(input_ids)

        # 5. Insert vision embeddings at image token positions
        inputs_embeds, attention_mask = self.insert_vision_embeddings(
            text_embeddings,
            projected_vision_embeddings.to(
                device=text_embeddings.device, dtype=text_embeddings.dtype
            ),
            input_ids,
        )

        # 6. Generate using the LLM with prepared embeddings
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    def compute_choice_log_likelihood(
        self,
        slide_latents: torch.Tensor,
        question: str,
        choice_text: str,
    ) -> dict:
        """
        Compute the length-normalized log-likelihood for a specific answer choice.

        This method constructs a conversation with the question and a candidate answer,
        runs a forward pass, and computes the log probability of the choice tokens
        given the image and question context.

        Args:
            slide_latents: Vision embeddings [1, num_latents, embedding_dim]
            question: The question text
            choice_text: The answer choice text to evaluate

        Returns:
            Dictionary with keys:
                - 'nll': Negative log-likelihood (sum of -log P)
                - 'normalized_nll': NLL divided by number of tokens
                - 'perplexity': exp(normalized_nll)
                - 'num_tokens': Number of tokens in the choice
        """
        llm_device = next(self.llm.parameters()).device

        # Ensure input tensor has correct dtype
        slide_latents = slide_latents.to(
            dtype=next(self.projection_layer.parameters()).dtype
        )

        # 1. Build conversation with question (user) and choice (assistant)
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": choice_text},
        ]

        # 2. Process conversation using the same template construction as forward()
        # Stage 1: Modify conversation structure for the templater
        processed_conversations = []
        modified_conv = []
        found_first_user = False

        for turn in conversation:
            if not found_first_user and turn["role"] == "user":
                # Reformat content from a string to a list of dicts
                original_content = turn["content"]
                modified_content = [
                    {"type": "image"},
                    {"type": "text", "text": original_content},
                ]
                modified_turn = {"role": "user", "content": modified_content}
                modified_conv.append(modified_turn)
                found_first_user = True
            else:
                modified_conv.append(turn)
        processed_conversations.append(modified_conv)

        # Stage 2a: Apply template to get string with simple placeholder
        templated_text = self.processor.tokenizer.apply_chat_template(
            processed_conversations,
            tokenize=False,
            add_generation_prompt=False,  # We already have the assistant response
        )

        # Stage 2b: Expand the simple image placeholder with the full sequence
        final_text = [
            s.replace("<start_of_image>", self.processor.full_image_sequence)
            for s in templated_text
        ]

        # Stage 2c: Tokenize the final, complete string
        text_inputs = self.processor.tokenizer(
            final_text,
            return_tensors="pt",
            padding=True,
        ).to(llm_device)
        input_ids = text_inputs.input_ids

        # 3. Tokenize the choice text separately to identify its tokens
        choice_tokens = self.processor.tokenizer(
            choice_text,
            return_tensors="pt",
            add_special_tokens=False,  # Critical: no special tokens
        ).input_ids[0].to(llm_device)

        # 4. Project vision embeddings
        projected_vision_embeddings = self.projection_layer(slide_latents)

        # 5. Get text embeddings from the LLM
        text_embeddings = self.llm.get_input_embeddings()(input_ids)

        # 6. Insert vision embeddings at image token positions
        inputs_embeds, attention_mask = self.insert_vision_embeddings(
            text_embeddings,
            projected_vision_embeddings.to(
                device=text_embeddings.device, dtype=text_embeddings.dtype
            ),
            input_ids,
        )

        # 7. Forward pass to get logits (inference mode, no gradients)
        with torch.inference_mode():
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )

        # 8. Extract logits
        logits = outputs.logits[0]  # [seq_len, vocab_size]

        # 9. Find where choice tokens appear in the full sequence
        # Search for the last occurrence (should be in assistant turn)
        full_ids = input_ids[0]
        choice_len = len(choice_tokens)
        start_idx = -1

        for i in range(len(full_ids) - choice_len + 1):
            if torch.equal(full_ids[i : i + choice_len], choice_tokens):
                start_idx = i
                # Keep searching to find the last occurrence

        if start_idx == -1:
            raise ValueError(
                f"Choice tokens for '{choice_text}' not found in sequence. "
                f"This may indicate a tokenization mismatch."
            )

        # 10. Compute log probabilities for choice tokens
        # For causal LM: logits[i-1] predicts token[i]
        # We need logits at positions [start_idx-1 : start_idx+choice_len-1]
        log_probs = torch.nn.functional.log_softmax(
            logits[start_idx - 1 : start_idx + choice_len - 1],
            dim=-1,
        )

        # 11. Extract log probability for each actual choice token
        token_log_probs = log_probs[
            torch.arange(choice_len),
            choice_tokens,
        ]

        # 12. Compute metrics
        nll = -token_log_probs.sum().item()
        num_tokens = choice_len
        normalized_nll = nll / num_tokens
        perplexity = torch.exp(torch.tensor(normalized_nll)).item()

        return {
            "nll": nll,
            "normalized_nll": normalized_nll,
            "perplexity": perplexity,
            "num_tokens": num_tokens,
        }

    def evaluate_multiple_choice(
        self,
        slide_latents: torch.Tensor,
        question: str,
        choices: dict[str, str],
    ) -> tuple[str, dict[str, float]]:
        """
        Evaluate multiple-choice question using length-normalized log-likelihood ranking.

        For each choice, computes the normalized perplexity of the choice text given
        the image and question. Returns the choice with the lowest perplexity (most likely).

        Args:
            slide_latents: Vision embeddings [1, num_latents, embedding_dim]
            question: The question text
            choices: Dictionary mapping choice letters to choice text
                    (e.g., {"A": "option1", "B": "option2", ...})

        Returns:
            Tuple of (predicted_letter, normalized_nll_dict) where:
                - predicted_letter: The letter of the best choice (lowest perplexity)
                - normalized_nll_dict: Dict mapping each choice letter to its normalized NLL
        """
        normalized_nlls = {}

        for choice_letter, choice_text in choices.items():
            result = self.compute_choice_log_likelihood(
                slide_latents, question, choice_text
            )
            # Use normalized NLL for fair comparison across different length options
            normalized_nlls[choice_letter] = result["normalized_nll"]

        # Return choice with lowest normalized NLL (equivalently, lowest perplexity)
        best_choice = min(normalized_nlls, key=normalized_nlls.get)
        return best_choice, normalized_nlls

    def debug_choice_evaluation(
        self,
        slide_latents: torch.Tensor,
        question: str,
        choice_text: str,
        output_file: str = None,
    ) -> str:
        """
        Debug version of compute_choice_log_likelihood that dumps all intermediate states.

        This method shows exactly how the template is constructed and how log probabilities
        are computed, making it easy to verify correctness.

        Args:
            slide_latents: Vision embeddings [1, num_latents, embedding_dim]
            question: The question text
            choice_text: The answer choice text to evaluate
            output_file: Optional file path to save debug output (if None, returns as string)

        Returns:
            String containing detailed debug information
        """
        llm_device = next(self.llm.parameters()).device
        slide_latents = slide_latents.to(
            dtype=next(self.projection_layer.parameters()).dtype
        )

        debug_lines = []
        debug_lines.append("=" * 100)
        debug_lines.append("PERPLEXITY EVALUATION DEBUG - INTERMEDIATE STATES")
        debug_lines.append("=" * 100)

        # ========== STEP 1: Input ==========
        debug_lines.append("\n" + "=" * 100)
        debug_lines.append("STEP 1: INPUT")
        debug_lines.append("=" * 100)
        debug_lines.append(f"\nQuestion:\n  {question}")
        debug_lines.append(f"\nChoice Text:\n  {choice_text}")
        debug_lines.append(f"\nSlide Latents Shape: {slide_latents.shape}")

        # ========== STEP 2: Conversation Structure ==========
        debug_lines.append("\n" + "=" * 100)
        debug_lines.append("STEP 2: CONVERSATION STRUCTURE")
        debug_lines.append("=" * 100)

        # Original conversation
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": choice_text},
        ]
        debug_lines.append("\nOriginal Conversation:")
        for i, turn in enumerate(conversation):
            debug_lines.append(f"  Turn {i}: {turn}")

        # Modified conversation (Stage 1)
        modified_conv = []
        found_first_user = False
        for turn in conversation:
            if not found_first_user and turn["role"] == "user":
                original_content = turn["content"]
                modified_content = [
                    {"type": "image"},
                    {"type": "text", "text": original_content},
                ]
                modified_turn = {"role": "user", "content": modified_content}
                modified_conv.append(modified_turn)
                found_first_user = True
            else:
                modified_conv.append(turn)

        debug_lines.append("\nModified Conversation (with image token placeholder):")
        for i, turn in enumerate(modified_conv):
            debug_lines.append(f"  Turn {i}:")
            debug_lines.append(f"    Role: {turn['role']}")
            debug_lines.append(f"    Content: {turn['content']}")

        # ========== STEP 3: Template Application ==========
        debug_lines.append("\n" + "=" * 100)
        debug_lines.append("STEP 3: TEMPLATE APPLICATION")
        debug_lines.append("=" * 100)

        processed_conversations = [modified_conv]
        templated_text = self.processor.tokenizer.apply_chat_template(
            processed_conversations,
            tokenize=False,
            add_generation_prompt=False,
        )

        debug_lines.append("\nTemplated Text (Stage 2a - with <start_of_image> placeholder):")
        debug_lines.append(f"  {repr(templated_text[0])}")

        # ========== STEP 4: Image Token Expansion ==========
        debug_lines.append("\n" + "=" * 100)
        debug_lines.append("STEP 4: IMAGE TOKEN EXPANSION")
        debug_lines.append("=" * 100)

        debug_lines.append(f"\nImage placeholder: <start_of_image>")
        debug_lines.append(f"Full image sequence length: {len(self.processor.full_image_sequence)} characters")
        debug_lines.append(f"Number of <image> tokens: {self.processor.full_image_sequence.count('<image>')}")

        final_text = [
            s.replace("<start_of_image>", self.processor.full_image_sequence)
            for s in templated_text
        ]

        # Show abbreviated version (full version would be too long)
        preview_len = 500
        full_text_preview = final_text[0][:preview_len] + "..." if len(final_text[0]) > preview_len else final_text[0]
        debug_lines.append(f"\nFinal Text (Stage 2b - with expanded image tokens) [first {preview_len} chars]:")
        debug_lines.append(f"  {repr(full_text_preview)}")
        debug_lines.append(f"\nFull text length: {len(final_text[0])} characters")

        # ========== STEP 5: Tokenization ==========
        debug_lines.append("\n" + "=" * 100)
        debug_lines.append("STEP 5: TOKENIZATION")
        debug_lines.append("=" * 100)

        text_inputs = self.processor.tokenizer(
            final_text,
            return_tensors="pt",
            padding=True,
        ).to(llm_device)
        input_ids = text_inputs.input_ids

        debug_lines.append(f"\nFull sequence shape: {input_ids.shape}")
        debug_lines.append(f"Full sequence length: {input_ids.shape[1]} tokens")

        # Show token IDs and decoded text
        debug_lines.append("\nFull Token Sequence (first 50 and last 50 tokens):")
        full_ids = input_ids[0]

        for idx in list(range(min(50, len(full_ids)))) + (list(range(len(full_ids)-50, len(full_ids))) if len(full_ids) > 100 else []):
            if idx == 50 and len(full_ids) > 100:
                debug_lines.append(f"  ... ({len(full_ids) - 100} tokens omitted) ...")
                continue
            if idx >= 50 and idx < len(full_ids) - 50:
                continue
            token_id = full_ids[idx].item()
            token_text = self.processor.tokenizer.decode([token_id])
            debug_lines.append(f"  [{idx:4d}] ID={token_id:6d}  Text={repr(token_text)}")

        # ========== STEP 6: Choice Token Identification ==========
        debug_lines.append("\n" + "=" * 100)
        debug_lines.append("STEP 6: CHOICE TOKEN IDENTIFICATION")
        debug_lines.append("=" * 100)

        choice_tokens = self.processor.tokenizer(
            choice_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0].to(llm_device)

        debug_lines.append(f"\nChoice text: {repr(choice_text)}")
        debug_lines.append(f"Number of choice tokens: {len(choice_tokens)}")
        debug_lines.append("\nChoice tokens:")
        for i, token_id in enumerate(choice_tokens):
            token_text = self.processor.tokenizer.decode([token_id])
            debug_lines.append(f"  [{i:2d}] ID={token_id.item():6d}  Text={repr(token_text)}")

        # Find choice tokens in sequence
        choice_len = len(choice_tokens)
        start_idx = -1
        all_matches = []

        for i in range(len(full_ids) - choice_len + 1):
            if torch.equal(full_ids[i : i + choice_len], choice_tokens):
                all_matches.append(i)
                start_idx = i

        debug_lines.append(f"\nSearching for choice tokens in full sequence...")
        if all_matches:
            debug_lines.append(f"Found {len(all_matches)} match(es) at position(s): {all_matches}")
            debug_lines.append(f"Using last match at position: {start_idx}")
        else:
            debug_lines.append("ERROR: Choice tokens not found in sequence!")
            debug_output = "\n".join(debug_lines)
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(debug_output)
            return debug_output

        # Show context around choice tokens
        debug_lines.append(f"\nContext around choice tokens (positions {start_idx} to {start_idx + choice_len - 1}):")
        context_start = max(0, start_idx - 10)
        context_end = min(len(full_ids), start_idx + choice_len + 10)

        for idx in range(context_start, context_end):
            token_id = full_ids[idx].item()
            token_text = self.processor.tokenizer.decode([token_id])
            is_choice = start_idx <= idx < start_idx + choice_len
            marker = " ← CHOICE TOKEN" if is_choice else ""
            debug_lines.append(f"  [{idx:4d}] ID={token_id:6d}  Text={repr(token_text)}{marker}")

        # ========== STEP 7: Forward Pass & Log Probabilities ==========
        debug_lines.append("\n" + "=" * 100)
        debug_lines.append("STEP 7: FORWARD PASS & LOG PROBABILITY COMPUTATION")
        debug_lines.append("=" * 100)

        # Project and insert vision embeddings
        projected_vision_embeddings = self.projection_layer(slide_latents)
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask = self.insert_vision_embeddings(
            text_embeddings,
            projected_vision_embeddings.to(
                device=text_embeddings.device, dtype=text_embeddings.dtype
            ),
            input_ids,
        )

        debug_lines.append(f"\nProjected vision embeddings shape: {projected_vision_embeddings.shape}")
        debug_lines.append(f"Inputs embeds shape: {inputs_embeds.shape}")
        debug_lines.append(f"Attention mask shape: {attention_mask.shape}")

        # Forward pass
        with torch.inference_mode():
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )

        logits = outputs.logits[0]
        debug_lines.append(f"\nLogits shape: {logits.shape}")

        # Compute log probabilities
        debug_lines.append(f"\nComputing log probabilities for choice tokens...")
        debug_lines.append(f"Note: For causal LM, logits[i-1] predicts token[i]")
        debug_lines.append(f"Using logits at positions [{start_idx-1}:{start_idx+choice_len-1}]")

        log_probs = torch.nn.functional.log_softmax(
            logits[start_idx - 1 : start_idx + choice_len - 1],
            dim=-1,
        )

        token_log_probs = log_probs[
            torch.arange(choice_len),
            choice_tokens,
        ]

        debug_lines.append("\nPer-token log probabilities:")
        debug_lines.append(f"{'Token':<4} {'ID':<8} {'Text':<30} {'Logit Pos':<12} {'Log Prob':<12} {'Prob':<12}")
        debug_lines.append("-" * 100)

        for i in range(choice_len):
            token_id = choice_tokens[i].item()
            token_text = self.processor.tokenizer.decode([token_id])
            logit_pos = start_idx - 1 + i
            log_prob = token_log_probs[i].item()
            prob = torch.exp(token_log_probs[i]).item()
            debug_lines.append(
                f"{i:<4} {token_id:<8} {repr(token_text):<30} {logit_pos:<12} "
                f"{log_prob:<12.4f} {prob:<12.6f}"
            )

        # ========== STEP 8: Final Metrics ==========
        debug_lines.append("\n" + "=" * 100)
        debug_lines.append("STEP 8: FINAL METRICS")
        debug_lines.append("=" * 100)

        nll = -token_log_probs.sum().item()
        num_tokens = choice_len
        normalized_nll = nll / num_tokens
        perplexity = torch.exp(torch.tensor(normalized_nll)).item()

        debug_lines.append(f"\nSum of log probabilities: {token_log_probs.sum().item():.6f}")
        debug_lines.append(f"Negative Log-Likelihood (NLL): {nll:.6f}")
        debug_lines.append(f"Number of tokens: {num_tokens}")
        debug_lines.append(f"Normalized NLL (NLL / num_tokens): {normalized_nll:.6f}")
        debug_lines.append(f"Perplexity (exp(normalized_nll)): {perplexity:.6f}")

        debug_lines.append("\n" + "=" * 100)
        debug_lines.append("END OF DEBUG OUTPUT")
        debug_lines.append("=" * 100)

        # Compile output
        debug_output = "\n".join(debug_lines)

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(debug_output)
            logger.info(f"Debug output saved to {output_file}")

        return debug_output
