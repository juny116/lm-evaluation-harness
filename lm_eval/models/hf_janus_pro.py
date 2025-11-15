from __future__ import annotations

import logging
from typing import Any

import torch
import transformers

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

eval_logger = logging.getLogger(__name__)


@register_model("hf-janus-pro")
class HFJanusProLM(HFLM):
    """
    Hugging Face backed loader for Janus-Pro checkpoints.

    Janus-Pro is a multimodal model (text, image generation/understanding) that uses
    a unified transformer architecture. This wrapper restricts evaluation to pure 
    text-to-text generation for compatibility with standard LM evaluation flows.
    
    Key differences from standard HFLM:
    - Uses VLChatProcessor for text processing
    - Uses MultiModalityCausalLM model class
    - Accesses the language_model component for text-only generation
    """

    AUTO_MODEL_CLASS = None  # Will be set to MultiModalityCausalLM
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
            tokenizer: Ignored for Janus-Pro, VLChatProcessor is loaded from pretrained path.
            think_end_token: Optional marker to strip thinking content from output.
            enable_thinking: For chat template rendering.
            trust_remote_code: Required for Janus-Pro models.
            **kwargs: Additional arguments passed to HFLM.__init__
        """
        if "trust_remote_code" in kwargs:
            trust_remote_code = kwargs.pop("trust_remote_code")

        if trust_remote_code is None:
            trust_remote_code = True

        # Set the model class for Janus-Pro (MultiModalityCausalLM)
        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            self.AUTO_MODEL_CLASS = MultiModalityCausalLM
            self._vl_chat_processor_class = VLChatProcessor
        except ImportError:
            eval_logger.warning(
                "Janus models not found. Please install janus package."
            )
            # Fallback to AutoModelForCausalLM if Janus not available
            self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            self._vl_chat_processor_class = None

        # Load VLChatProcessor from pretrained path to get tokenizer
        if tokenizer is None and isinstance(pretrained, str):
            try:
                if self._vl_chat_processor_class is not None:
                    eval_logger.info(f"Loading VLChatProcessor from {pretrained}")
                    vl_chat_processor = self._vl_chat_processor_class.from_pretrained(
                        pretrained,
                        trust_remote_code=trust_remote_code,
                    )
                    # Store processor for potential future use
                    self._vl_chat_processor = vl_chat_processor
                    # Extract tokenizer from processor
                    tokenizer = vl_chat_processor.tokenizer
                    eval_logger.info("Extracted tokenizer from VLChatProcessor")
                else:
                    eval_logger.warning("VLChatProcessor not available, loading tokenizer directly")
                    self._vl_chat_processor = None
            except Exception as e:
                eval_logger.warning(f"Failed to load VLChatProcessor: {e}. Will use default tokenizer loading.")
                self._vl_chat_processor = None

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
                eval_logger.info("Removed max_new_tokens from Janus-Pro generation_config")
        
        # Also check language_model's generation_config if it exists
        if hasattr(self.model, 'language_model'):
            if hasattr(self.model.language_model, 'generation_config') and self.model.language_model.generation_config is not None:
                if hasattr(self.model.language_model.generation_config, 'max_new_tokens'):
                    self.model.language_model.generation_config.max_new_tokens = None
                    eval_logger.info("Removed max_new_tokens from language_model generation_config")

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for Janus-Pro model.
        
        For text-only evaluation, we use the language_model component directly.
        """
        # Access the language model component for text-only forward pass
        language_model = getattr(self.model, 'language_model', self.model)
        
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
                outputs = language_model(**model_kwargs)
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
                    outputs = language_model(**model_kwargs)
                    if hasattr(outputs, "logits"):
                        return outputs.logits
                    if isinstance(outputs, torch.Tensor):
                        return outputs
                raise

            raise RuntimeError("Janus-Pro model call did not yield logits.")

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: list[str],
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate method for Janus-Pro.
        
        For text-only generation, we use the language_model.generate() method directly
        with input_ids, similar to the official Janus-Pro usage pattern.
        """
        # Access the language model for generation
        language_model = getattr(self.model, 'language_model', self.model)
        
        # Janus-Pro's language_model.generate expects input_ids
        # Remove 'attention_mask' from generation_kwargs as it will be passed separately
        attn_mask = generation_kwargs.pop('attention_mask', None)
        
        # Build stopping criteria
        from lm_eval.models.utils import stop_sequences_criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        
        # Set special tokens for generation
        generation_kwargs.setdefault('pad_token_id', self.tokenizer.eos_token_id)
        generation_kwargs.setdefault('bos_token_id', self.tokenizer.bos_token_id)
        generation_kwargs.setdefault('eos_token_id', self.tokenizer.eos_token_id)
        generation_kwargs.setdefault('use_cache', True)
        
        # Handle temperature and sampling
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")
        
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False
        
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            return language_model.generate(
                input_ids=context,
                attention_mask=attn_mask,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                **generation_kwargs,
            )

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Apply Janus-Pro's chat template using VLChatProcessor.apply_sft_template_for_multi_turn_prompts.
        
        Janus-Pro format (from test results):
        - Format: <|User|>: content\n\n<|Assistant|>: content<｜end▁of▁sentence｜><|User|>: ...
        - Role tokens: <|User|> and <|Assistant|> (with colon and space after)
        - No <|System|> token - system messages prepended as plain text
        - Between turns: EOS token <｜end▁of▁sentence｜>
        """
        # DEBUG: Print input chat history
        # eval_logger.info(f"[DEBUG] Input chat_history: {chat_history}")
        
        # Convert LM Eval standard roles to Janus-Pro conversation format
        janus_conversations = []
        system_prompt = ""
        
        for message in chat_history:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Handle system messages separately
            if role.lower() == "system":
                # Janus doesn't have <|System|> token, prepend to system_prompt
                # Strip trailing newlines to avoid excessive spacing
                system_prompt = content.rstrip()
                continue
            
            # Map to Janus roles - keep the exact format expected by apply_sft_template_for_multi_turn_prompts
            if role.lower() in ["user", "human"]:
                janus_role = "<|User|>"
            elif role.lower() in ["assistant", "ai", "bot"]:
                janus_role = "<|Assistant|>"
            elif role.startswith("<|"):
                # Already in Janus format
                janus_role = role
            else:
                # Unknown role, default to User
                eval_logger.warning(f"Unknown role '{role}', defaulting to <|User|>")
                janus_role = "<|User|>"
            
            janus_conversations.append({
                "role": janus_role,
                "content": content
            })
        
        # DEBUG: Print converted conversations
        # eval_logger.info(f"[DEBUG] Converted conversations: {janus_conversations}")
        # eval_logger.info(f"[DEBUG] System prompt: '{system_prompt}'")
        
        # Try to use VLChatProcessor.apply_sft_template_for_multi_turn_prompts if available
        if hasattr(self, '_vl_chat_processor') and self._vl_chat_processor is not None:
            try:
                # If add_generation_prompt is True and last message is not Assistant with empty content,
                # we need to add an empty Assistant message
                conversations_for_template = janus_conversations.copy()
                if add_generation_prompt:
                    if not conversations_for_template or conversations_for_template[-1]['role'] != "<|Assistant|>" or conversations_for_template[-1]['content']:
                        conversations_for_template.append({
                            "role": "<|Assistant|>",
                            "content": ""
                        })
                
                sft_format = self._vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=conversations_for_template,
                    sft_format=self._vl_chat_processor.sft_format,
                    system_prompt=system_prompt,
                )
                # DEBUG: Print VLChatProcessor result
                # eval_logger.info(f"[DEBUG] VLChatProcessor result:\n{repr(sft_format)}")
                return sft_format
            except Exception as e:
                eval_logger.warning(
                    f"Failed to use VLChatProcessor.apply_sft_template_for_multi_turn_prompts: {e}. Using manual formatting."
                )
        
        # Manual formatting fallback (mimics exact Janus conversation format from tests)
        # Format: <|User|>: content\n\n<|Assistant|>: content<｜end▁of▁sentence｜>...
        formatted_text = ""
        
        # Add system prompt as initial context if exists
        if system_prompt:
            formatted_text += f"{system_prompt}\n\n"
        
        for i, conv in enumerate(janus_conversations):
            role = conv['role']
            content = conv['content']
            
            # Add role with colon and space
            formatted_text += f"{role}: {content}"
            
            # Add proper spacing/separator
            if i < len(janus_conversations) - 1:
                # Not the last message
                if conv['role'] == "<|Assistant|>" and content:
                    # After assistant response with content, add EOS token
                    formatted_text += f"{self.tokenizer.eos_token}"
                else:
                    # Just add double newline
                    formatted_text += "\n\n"
            else:
                # Last message
                if not add_generation_prompt:
                    # Don't add anything after last message
                    pass
                elif conv['role'] == "<|Assistant|>":
                    # Already ends with assistant role
                    if content:
                        formatted_text += f"{self.tokenizer.eos_token}"
                else:
                    # Last message is user, add assistant prompt
                    formatted_text += "\n\n<|Assistant|>:"
        
        # If add_generation_prompt and last message was assistant with empty content, add colon
        if add_generation_prompt and janus_conversations and janus_conversations[-1]['role'] == "<|Assistant|>" and not janus_conversations[-1]['content']:
            if not formatted_text.endswith(':'):
                formatted_text += ":"
        
        # DEBUG: Print manual result
        # eval_logger.info(f"[DEBUG] Manual format result:\n{repr(formatted_text)}")
        return formatted_text
