from __future__ import annotations

import logging
from typing import Any

import torch
import transformers

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

eval_logger = logging.getLogger(__name__)


@register_model("hf-qwen-omni")
class HFQwenOmniLM(HFLM):
    """
    Hugging Face backed loader for Qwen2.5-Omni checkpoints.

    The underlying checkpoint is multimodal (text, audio, vision, video), but this
    wrapper intentionally restricts evaluation to pure text-to-text generation so it
    plugs into the standard LM evaluation flow without extra media handling.
    """

    # 기본값은 풀 Omni 합성 모델. __init__에서 use_thinker에 따라 교체 가능.
    AUTO_MODEL_CLASS = transformers.Qwen2_5OmniForConditionalGeneration
    MULTIMODAL = False

    def __init__(
        self,
        pretrained: str | transformers.PreTrainedModel,
        disable_talker: bool = True,
        use_thinker: bool | None = True,
        return_audio: bool = False,
        use_audio_in_video: bool = False,
        enable_thinking: bool = True,
        think_end_token: str | int | None = None,
        trust_remote_code: bool | None = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            disable_talker: Call `model.disable_talker()` after loading so the speech
                decoder is disabled and the model behaves like a pure text LM.
            return_audio: Whether to request audio from `model.generate`. This should
                stay False for LM eval so the base class always receives text tokens.
            use_audio_in_video: Mirrors the upstream API flag but defaults to False
                because we never feed audio extracted from videos in text-only evals.
            enable_thinking: For chat template rendering. Qwen Omni exposes thinking
                tokens that improve reasoning quality when enabled.
            think_end_token: Optional marker used to strip the hidden "thinking"
                content off of the decoded continuation (string token or token id).
        """
        if "trust_remote_code" in kwargs:
            trust_remote_code = kwargs.pop("trust_remote_code")

        if trust_remote_code is None:
            trust_remote_code = True

        # use_thinker 자동 감지: 명시 플래그 우선, 없으면 repo id나 config 문자열에 'Thinker' 포함 여부 판단
        if use_thinker is None:
            pretrained_name = pretrained.name_or_path if isinstance(pretrained, transformers.PreTrainedModel) else str(pretrained)
            lower_name = pretrained_name.lower()
            # 규칙: -thinker 포함, thinker 단어 포함, 또는 talker/오디오 비활성화 시 암시적 선택
            if ("thinker" in lower_name) or (disable_talker and "omni" in lower_name):
                use_thinker = True
            else:
                use_thinker = False

        if use_thinker:
            # Thinker 전용 모델 사용 (언어/추론 부분만)
            self.AUTO_MODEL_CLASS = transformers.Qwen2_5OmniThinkerForConditionalGeneration
        else:
            self.AUTO_MODEL_CLASS = transformers.Qwen2_5OmniForConditionalGeneration

        self.use_thinker = use_thinker

        super().__init__(
            pretrained=pretrained,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
            think_end_token=think_end_token,
            **kwargs,
        )
        self.return_audio = return_audio
        self.use_audio_in_video = use_audio_in_video
        self.disable_talker = disable_talker

        if self.disable_talker and hasattr(self.model, "disable_talker"):
            try:
                self.model.disable_talker()
            except Exception as err:  # pragma: no cover - best effort safeguard
                eval_logger.warning(
                    "Failed to disable talker on Qwen Omni model: %s", err
                )

    def _model_generate(
        self,
        context,
        max_length: int,
        stop: list[str],
        **generation_kwargs: Any,
    ):
        # Force text-only decoding so the HFLM base class always receives token IDs.
        # 풀 Omni 모델은 generate 시 return_audio / use_audio_in_video 매개변수를 지원하지만
        # Thinker 전용 모델(Qwen2_5OmniThinkerForConditionalGeneration)은 지원하지 않으므로 제거 필요.
        if not self.use_thinker:
            generation_kwargs.setdefault("return_audio", self.return_audio)
            generation_kwargs.setdefault("use_audio_in_video", self.use_audio_in_video)
        else:
            # 사용자가 혹시 직접 넣었다면 제거하여 transformers generate 검증 통과
            generation_kwargs.pop("return_audio", None)
            generation_kwargs.pop("use_audio_in_video", None)

        generations = super()._model_generate(
            context=context,
            max_length=max_length,
            stop=stop,
            **generation_kwargs,
        )

        # If upstream ever returns (text_ids, audio) ensure we keep text ids only.
        if isinstance(generations, (tuple, list)):
            generations = generations[0]

        return generations

    def _model_call(
        self,
        inps,
        attn_mask=None,
        labels=None,
    ):
        """
        Qwen Omni's `transformers` class is not one of the canonical AutoModel classes
        that `HFLM` whitelists, so we bypass the parent assertion and call it directly.
        """

        # Thinker 또는 풀 Omni 모델인지에 따라 forward 경로 분리
        # 1) use_thinker=True: 모델 자체가 ThinkerForConditionalGeneration
        # 2) use_thinker=False + full omni: 내부 self.model.thinker 서브모델을 텍스트 전용으로 사용
        #    (self.model이 Qwen2_5OmniForConditionalGeneration 인스턴스)

        if self.use_thinker:
            target_model = self.model  # 직접 thinker 클래스
        else:
            # 풀 모델이면 thinker 서브모듈 존재 가정
            target_model = getattr(self.model, "thinker", self.model)

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
                outputs = target_model(**model_kwargs)
                if isinstance(outputs, torch.Tensor):
                    return outputs  # logits directly
                if hasattr(outputs, "logits"):
                    return outputs.logits
            except TypeError as e:
                # full omni composite일 때 thinker 서브모델 직접 접근 시도
                if not self.use_thinker and hasattr(self.model, "thinker"):
                    try:
                        thinker_out = self.model.thinker(**model_kwargs)
                        if hasattr(thinker_out, "logits"):
                            return thinker_out.logits
                        if isinstance(thinker_out, torch.Tensor):
                            return thinker_out
                    except Exception:
                        pass
                if "_forward_unimplemented" in str(e):
                    # 단순 임베딩 경로 수동 구축 (드물게 필요)
                    text_model = (
                        getattr(target_model, "language_model", None)
                        or getattr(target_model, "text_model", None)
                        or getattr(target_model, "model", None)
                        or getattr(target_model, "transformer", None)
                    )
                    lm_head = getattr(target_model, "lm_head", None)
                    if text_model is None or lm_head is None:
                        raise
                    text_outputs = text_model(input_ids=inps, attention_mask=attn_mask)
                    hidden = (
                        text_outputs[0]
                        if isinstance(text_outputs, tuple)
                        else getattr(text_outputs, "last_hidden_state", text_outputs)
                    )
                    return lm_head(hidden)
                raise
            raise RuntimeError("Qwen Omni model call did not yield logits.")
