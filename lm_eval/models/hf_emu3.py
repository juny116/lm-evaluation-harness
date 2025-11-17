from __future__ import annotations

import logging
from typing import Any

import torch
import transformers

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

eval_logger = logging.getLogger(__name__)


@register_model("hf-emu3")
class HFEmu3LM(HFLM):
    """
    Hugging Face backed loader for Emu3-Chat checkpoints.

    Emu3 is a multimodal model (text, image, video) that uses next-token prediction.
    This wrapper restricts evaluation to pure text-to-text generation for compatibility
    with standard LM evaluation flows.
    
    Key differences from standard HFLM:
    - Uses AutoProcessor instead of AutoTokenizer for text processing
    - Supports Emu3ForConditionalGeneration model class
    """

    AUTO_MODEL_CLASS = None  # Will be set to Emu3ForConditionalGeneration
    MULTIMODAL = False

    def __init__(
        self,
        pretrained: str | transformers.PreTrainedModel,
        tokenizer: str
        | transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast
        | None = None,
        think_end_token: str | int | None = None,
        enable_thinking: bool | None = None,
        trust_remote_code: bool | None = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            pretrained: Model name or path, or a PreTrainedModel instance.
            tokenizer: Ignored for Emu3, processor is loaded from pretrained path.
            think_end_token: Optional marker to strip thinking content from output.
            enable_thinking: For chat template rendering.
            trust_remote_code: Required for Emu3 models.
            **kwargs: Additional arguments passed to HFLM.__init__
        """
        if "trust_remote_code" in kwargs:
            trust_remote_code = kwargs.pop("trust_remote_code")

        if trust_remote_code is None:
            trust_remote_code = True

        # Set the model class for Emu3
        try:
            self.AUTO_MODEL_CLASS = transformers.Emu3ForConditionalGeneration
        except AttributeError:
            eval_logger.warning(
                "Emu3ForConditionalGeneration not found in transformers. "
                "Please ensure you have transformers >= 4.48.0 installed."
            )
            # Fallback to AutoModelForCausalLM if Emu3 class not available
            self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

        # Load processor from pretrained path to get tokenizer
        if tokenizer is None and isinstance(pretrained, str):
            try:
                eval_logger.info(f"Loading AutoProcessor from {pretrained}")
                processor = transformers.AutoProcessor.from_pretrained(
                    pretrained,
                    trust_remote_code=trust_remote_code,
                )
                # Store processor for chat template
                self._processor = processor
                # Extract tokenizer from processor
                tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                eval_logger.info("Extracted tokenizer from processor")
            except Exception as e:
                eval_logger.warning(f"Failed to load processor: {e}. Will use default tokenizer loading.")
                self._processor = None
        # set tokenizer template to processor template
        tokenizer.chat_template = getattr(self._processor, 'chat_template', None)
        super().__init__(
            pretrained=pretrained,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
            think_end_token=think_end_token,
            **kwargs,
        )
        
        # Remove max_new_tokens from generation_config to avoid conflicts with max_length
        if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
            if hasattr(self.model.generation_config, 'max_new_tokens'):
                self.model.generation_config.max_new_tokens = None
                eval_logger.info("Removed max_new_tokens from Emu3 generation_config")

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for Emu3 model.
        
        Emu3ForConditionalGeneration has a similar interface to standard causal LMs
        but we handle it explicitly to ensure compatibility.
        """
        model_kwargs: dict[str, Any] = {"input_ids": inps}
        if attn_mask is not None:
            model_kwargs["attention_mask"] = attn_mask
        if labels is not None:
            model_kwargs["labels"] = labels

        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ),
        ):
            try:
                outputs = self.model(**model_kwargs)
                if isinstance(outputs, torch.Tensor):
                    return outputs
                if hasattr(outputs, "logits"):
                    return outputs.logits
            except TypeError as e:
                # Handle potential signature mismatches
                eval_logger.warning(f"Model call failed with TypeError: {e}")
                # Try without labels if that was the issue
                if "labels" in model_kwargs:
                    model_kwargs.pop("labels")
                    outputs = self.model(**model_kwargs)
                    if hasattr(outputs, "logits"):
                        return outputs.logits
                    if isinstance(outputs, torch.Tensor):
                        return outputs
                raise

            raise RuntimeError("Emu3 model call did not yield logits.")

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: list[str],
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate method for Emu3.
        
        Uses standard HuggingFace generate interface. Emu3 supports text-only
        generation when no image inputs are provided.
        """
        return super()._model_generate(
            context=context,
            max_length=max_length,
            stop=stop,
            **generation_kwargs,
        )

    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        """
        Override _encode_pair to handle Emu3 tokenizer's quirk with short continuations.
        
        Emu3's tokenizer can merge very short continuations (like ".") with the
        preceding context token, resulting in empty continuation_enc. We fix this
        by ensuring the continuation always has proper spacing.
        """
        # Handle trailing spaces in context (standard behavior)
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces:]
        
        # Emu3-specific fix: If continuation is very short and doesn't start with space,
        # prepend a space to prevent tokenizer from merging it with context
        if continuation and not continuation[0].isspace() and len(continuation.strip()) <= 3:
            continuation = " " + continuation
        
        # Use parent class logic
        if self.backend == "causal":
            whole_enc = self.tok_encode(context + continuation)
            context_enc = self.tok_encode(context)
            
            context_enc_len = len(context_enc)
            continuation_enc = whole_enc[context_enc_len:]
        else:
            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
        
        return context_enc, continuation_enc

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Apply Emu3's chat template.
        
        For text-only conversations, we construct messages in the format:
        [{"role": "user", "content": [{"type": "text", "text": "..."}]}]
        
        The processor's apply_chat_template handles the formatting.
        
        Note: Emu3 template uses {% generation %} tags which can cause issues with
        continue_final_message. We avoid using that parameter to prevent empty continuations.
        """
        # Use processor if available, otherwise fall back to tokenizer
        # template_handler = getattr(self, '_processor', None) or self.tokenizer
        template_handler = self.tokenizer
        if hasattr(template_handler, "apply_chat_template"):
            # Convert simple text messages to Emu3 format if needed
            formatted_messages = []
            for message in chat_history:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                # If content is already a list (multimodal format), use as-is
                if isinstance(content, list):
                    formatted_messages.append({"role": role, "content": content})
                else:
                    # Convert text to Emu3 expected format
                    formatted_messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": content}]
                    })
            
            try:
                # Use processor's apply_chat_template
                # IMPORTANT: Don't use continue_final_message with Emu3's {% generation %} tags
                # as it can cause the continuation to become empty during tokenization.
                # Override chat_template_args to exclude continue_final_message
                template_args = dict(self.chat_template_args)
                template_args.pop('continue_final_message', None)  # Remove if exists
                
                result = template_handler.apply_chat_template(
                    formatted_messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False,  # Return string, not tokens
                    **template_args,
                )
                return result
            except Exception as e:
                eval_logger.warning(
                    f"Failed to apply Emu3 chat template: {e}. Using fallback."
                )
                # Fallback to basic formatting
                return super().apply_chat_template(
                    chat_history, add_generation_prompt=add_generation_prompt
                )
        else:
            return super().apply_chat_template(
                chat_history, add_generation_prompt=add_generation_prompt
            )
