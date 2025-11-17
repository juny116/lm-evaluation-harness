from __future__ import annotations

import logging
from typing import Any

import torch
import transformers

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

eval_logger = logging.getLogger(__name__)


@register_model("hf-x-omni")
class HFXOmniLM(HFLM):
    """
    Hugging Face backed loader for X-Omni checkpoints.

    X-Omni is a unified discrete autoregressive model for both image and language modalities.
    This wrapper restricts evaluation to pure text-to-text generation for compatibility
    with standard LM evaluation flows.
    
    Key differences from standard HFLM:
    - Uses AutoModelForCausalLM with custom vision initialization
    - Supports text generation mode via model.set_generation_mode('text')
    - Uses model.mmdecode() for output decoding
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
    MULTIMODAL = False

    def __init__(
        self,
        pretrained: str | transformers.PreTrainedModel,
        tokenizer: str
        | transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast
        | None = None,
        flux_model_name_or_path: str | None = None,
        skip_vision_init: bool = True,
        think_end_token: str | int | None = None,
        enable_thinking: bool | None = None,
        trust_remote_code: bool | None = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            pretrained: Model name or path, or a PreTrainedModel instance.
            tokenizer: Tokenizer name or path. If None, loads from pretrained path.
            flux_model_name_or_path: Path to FLUX.1-dev model for vision initialization.
                If None or skip_vision_init=True, vision is not initialized (text-only mode).
            skip_vision_init: If True, skip vision initialization for text-only evaluation.
            think_end_token: Optional marker to strip thinking content from output.
            enable_thinking: For chat template rendering.
            trust_remote_code: Required for X-Omni models.
            **kwargs: Additional arguments passed to HFLM.__init__
        """
        if "trust_remote_code" in kwargs:
            trust_remote_code = kwargs.pop("trust_remote_code")

        if trust_remote_code is None:
            trust_remote_code = True

        self.flux_model_name_or_path = flux_model_name_or_path
        self.skip_vision_init = skip_vision_init

        super().__init__(
            pretrained=pretrained,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
            think_end_token=think_end_token,
            **kwargs,
        )

        # Initialize vision if needed (required for image generation/understanding)
        if not self.skip_vision_init and self.flux_model_name_or_path:
            try:
                eval_logger.info(f"Initializing X-Omni vision with FLUX model: {self.flux_model_name_or_path}")
                self.model.init_vision(self.flux_model_name_or_path)
                eval_logger.info("Vision initialized successfully")
            except Exception as e:
                eval_logger.warning(f"Failed to initialize vision: {e}. Continuing in text-only mode.")
        
        # Set generation mode to text-only
        if hasattr(self.model, 'set_generation_mode'):
            try:
                self.model.set_generation_mode('text')
                eval_logger.info("Set X-Omni generation mode to 'text'")
            except Exception as e:
                eval_logger.warning(f"Failed to set generation mode: {e}")

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for X-Omni model.
        
        X-Omni uses standard AutoModelForCausalLM interface, so we can use
        the parent class implementation directly.
        """
        return super()._model_call(inps=inps, attn_mask=attn_mask, labels=labels)

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: list[str],
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate method for X-Omni.
        
        Uses standard HuggingFace generate interface with text-only mode.
        The model's set_generation_mode('text') ensures text-only output.
        """
        # X-Omni uses <|im_end|> as EOS token
        if self.tokenizer.eos_token == '<|im_end|>' or '<|im_end|>' in str(self.tokenizer.eos_token):
            eos_token_id = self.tokenizer.encode('<|im_end|>')[0]
            generation_kwargs.setdefault('eos_token_id', eos_token_id)
        
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
        Override _encode_pair to handle X-Omni tokenizer's quirk with short continuations.
        
        Similar to Emu3, X-Omni's tokenizer can merge very short continuations with
        preceding context tokens, resulting in empty continuation_enc.
        """
        # Handle trailing spaces in context (standard behavior)
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        
        # X-Omni-specific fix: If continuation is very short and doesn't start with space,
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
        Apply X-Omni's chat template.
        
        X-Omni uses standard transformers chat template format similar to Qwen.
        The tokenizer.apply_chat_template() should handle the formatting correctly.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                # Use tokenizer's built-in chat template
                result = self.tokenizer.apply_chat_template(
                    chat_history,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False,
                    **self.chat_template_args,
                )
                return result
            except Exception as e:
                eval_logger.warning(
                    f"Failed to apply X-Omni chat template: {e}. Using fallback."
                )
                # Fallback to basic formatting
                return super().apply_chat_template(
                    chat_history, add_generation_prompt=add_generation_prompt
                )
        else:
            return super().apply_chat_template(
                chat_history, add_generation_prompt=add_generation_prompt
            )
