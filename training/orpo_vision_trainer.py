"""
VisionORPOTrainer - ORPO trainer with vision-language model support.

D-ORPO — Decision-Focused ORPO
Standard ORPO computes the odds ratio over uniformly weighted response tokens. However, in our generate-or-refuse setting, the critical decision concentrates in the first few tokens. We propose D-ORPO, which applies a higher weight to the initial K response tokens in both the NLL and OR objectives, directing the optimization signal toward the decision boundary.

Key differences from standard ORPOTrainer:
  1. _prepare_dataset returns raw data (tokenization is deferred to the DataCollator
     which also handles images).
  2. concatenated_forward injects pixel_values / image_grid_thw into the model call
     so the vision encoder actually sees the images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable, Dict, Any
from collections import defaultdict

from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BaseImageProcessor,
    FeatureExtractionMixin,
    ProcessorMixin,
    DataCollator,
    TrainerCallback,
)
from transformers.trainer_utils import EvalLoopOutput
from trl import ORPOTrainer, ORPOConfig

try:
    from unsloth.trainer import UnslothORPOTrainer, UnslothORPOConfig
    BASE_TRAINER = UnslothORPOTrainer
    BASE_CONFIG = UnslothORPOConfig
except ImportError:
    BASE_TRAINER = ORPOTrainer
    BASE_CONFIG = ORPOConfig

_VISION_KEYS = ("pixel_values", "image_grid_thw", "pixel_values_videos")


class VisionORPOTrainer(BASE_TRAINER):
    """ORPO Trainer that supports vision-language models (e.g. Qwen2.5-VL, Qwen3-VL)."""

    # ------------------------------------------------------------------
    # Skip ALL built-in tokenisation; our DataCollator handles it.
    # Unsloth's __init__ calls tokenize_row via dataset.map() BEFORE
    # _prepare_dataset, so both must be no-ops.
    # ------------------------------------------------------------------
    def _prepare_dataset(self, dataset, *args, **kwargs):
        return dataset

    def tokenize_row(self, feature, *args, **kwargs):
        return feature

    # ------------------------------------------------------------------
    # __init__: just forward everything to the parent, then flip the flag
    # that would otherwise override our custom collator.
    # ------------------------------------------------------------------
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        peft_config=None,
        compute_metrics=None,
        use_dorpo: bool = True,
        decision_tokens: int = 16,
        decision_weight: float = 3.0,
    ):
        if data_collator is None:
            raise ValueError(
                "VisionORPOTrainer requires a custom data_collator that "
                "handles image tokenisation (e.g. ORPOVisionDataCollator)."
            )

        shared = dict(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            compute_metrics=compute_metrics,
        )

        # Keep a reference — Unsloth's __init__ may overwrite self.data_collator
        _custom_collator = data_collator

        try:
            super().__init__(processing_class=processing_class, **shared)
        except TypeError:
            super().__init__(tokenizer=processing_class, **shared)

        # Force our vision collator back (Unsloth replaces it with a default one)
        self.data_collator = _custom_collator
        self.use_dpo_data_collator = False

        # D-ORPO config
        self.use_dorpo = use_dorpo
        self.decision_tokens = decision_tokens
        self.decision_weight = decision_weight

    # ------------------------------------------------------------------
    # Core override: single forward on concatenated chosen+rejected
    # with vision tensors injected into the model call.
    # ------------------------------------------------------------------
    def concatenated_forward(self, model, batch):
        # 1. Pop vision tensors (filter out None values)
        vision = {k: batch.pop(k) for k in _VISION_KEYS
                  if k in batch and batch[k] is not None}

        len_chosen = batch["chosen_labels"].shape[0]
        pad_id = getattr(self, "padding_value", 0)
        label_pad = getattr(self, "label_pad_token_id", -100)

        # 2. Pad chosen / rejected to the same length, then concatenate
        max_len = max(
            batch["chosen_input_ids"].shape[1],
            batch["rejected_input_ids"].shape[1],
        )

        def _pad(t, val):
            diff = max_len - t.shape[1]
            return F.pad(t, (0, diff), value=val) if diff > 0 else t

        concat_ids = torch.cat(
            [_pad(batch["chosen_input_ids"], pad_id),
             _pad(batch["rejected_input_ids"], pad_id)], dim=0)
        concat_mask = torch.cat(
            [_pad(batch["chosen_attention_mask"], 0),
             _pad(batch["rejected_attention_mask"], 0)], dim=0)
        concat_labels = torch.cat(
            [_pad(batch["chosen_labels"], label_pad),
             _pad(batch["rejected_labels"], label_pad)], dim=0)

        # mm_token_type_ids: same [batch, seq_len] layout as input_ids → pad + concat
        has_mm = "chosen_mm_token_type_ids" in batch
        if has_mm:
            concat_mm = torch.cat(
                [_pad(batch["chosen_mm_token_type_ids"], 0),
                 _pad(batch["rejected_mm_token_type_ids"], 0)], dim=0)

        # 3. Build model kwargs
        model_kwargs = {
            "input_ids": concat_ids,
            "attention_mask": concat_mask,
        }

        if has_mm:
            model_kwargs["mm_token_type_ids"] = concat_mm

        # Duplicate image tensors (chosen + rejected share the same images)
        for k, v in vision.items():
            model_kwargs[k] = torch.cat([v, v], dim=0)

        if getattr(self, "aux_loss_enabled", False):
            model_kwargs["output_router_logits"] = True

        # 4. Forward pass
        outputs = model(**model_kwargs)
        logits = outputs.logits

        # Vision models may return logits for image-token positions too
        if logits.shape[1] != concat_labels.shape[1]:
            logits = logits[:, -concat_labels.shape[1]:]

        # 5. Compute per-sequence average log-probs
        all_logps, n_toks = self._seq_logps(logits, concat_labels, label_pad)
        all_logps = all_logps / n_toks.clamp(min=1)

        # 6. NLL loss on chosen responses only
        nll_loss = self._nll(logits[:len_chosen], concat_labels[:len_chosen], label_pad)

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_logits = logits[:len_chosen]
        rejected_logits = logits[len_chosen:]

        result = (chosen_logps, rejected_logps,
                  chosen_logits, rejected_logits, nll_loss)
        if getattr(self, "aux_loss_enabled", False):
            result += (outputs.aux_loss,)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _seq_logps(self, logits, labels, ignore_id=-100):
        """Sum of log-probs and number of non-masked tokens per sequence.

        When ``self.use_dorpo`` is True (D-ORPO), the first
        ``self.decision_tokens`` response tokens receive
        ``self.decision_weight`` times more weight, focusing the
        optimisation signal on the decision boundary.  When False,
        all response tokens are weighted uniformly (standard ORPO).
        """
        s_logits = logits[:, :-1, :].contiguous()
        s_labels = labels[:, 1:].contiguous()
        per_tok = torch.gather(
            s_logits.float().log_softmax(-1), 2,
            s_labels.clamp(min=0).unsqueeze(2),
        ).squeeze(2)
        mask = s_labels != ignore_id

        if self.use_dorpo:
            response_start = mask.long().argmax(dim=1)  # [batch]
            positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
            offset = positions - response_start.unsqueeze(1)
            early = (offset >= 0) & (offset < self.decision_tokens)
            weights = torch.where(early, self.decision_weight, 1.0)
        else:
            weights = 1.0

        return (per_tok * mask * weights).sum(-1), (mask * weights).sum(-1)

    @staticmethod
    def _nll(logits, labels, ignore_id=-100):
        s_logits = logits[..., :-1, :].contiguous()
        s_labels = labels[..., 1:].contiguous()
        return nn.CrossEntropyLoss(ignore_index=ignore_id)(
            s_logits.view(-1, s_logits.size(-1)),
            s_labels.view(-1),
        )
