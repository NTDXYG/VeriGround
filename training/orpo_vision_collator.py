import math
import torch
from typing import List, Dict, Any, Optional
from PIL import Image
from dataclasses import dataclass, field

from unsloth_zoo.vision_utils import process_vision_info

def _dtype_from_model(model) -> torch.dtype:
    try:
        return model.get_input_embeddings().weight.dtype
    except Exception:
        return torch.float32


@dataclass
class ORPOVisionDataCollator:
    """
    ORPO data collator for vision-language models (Qwen2.5-VL / Qwen3-VL).

    Produces a batch dict with:
      chosen_input_ids / rejected_input_ids / chosen_labels / rejected_labels
      chosen_attention_mask / rejected_attention_mask
      pixel_values / image_grid_thw
      chosen_mm_token_type_ids / rejected_mm_token_type_ids   (Qwen3.5 M-RoPE)
    """

    model: Any
    tokenizer: Any
    max_length: int = 4096
    ignore_index: int = -100

    def __post_init__(self):
        self.processor = self.tokenizer
        self.model_dtype = _dtype_from_model(self.model)

        tok = getattr(self.processor, "tokenizer", self.processor)
        self.pad_token_id = tok.pad_token_id

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------
    def _load_image(self, image_path: str) -> Image.Image:
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            return Image.new("RGB", (224, 224), color=(128, 128, 128))

    def _extract_images(self, messages: List[Dict]) -> List[Image.Image]:
        """Extract PIL images from chat messages via process_vision_info or fallback."""
        if process_vision_info is not None:
            imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
            return imgs if imgs else []
        images = []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img = item.get("image")
                    if isinstance(img, Image.Image):
                        images.append(img)
        return images

    # ------------------------------------------------------------------
    # Text preparation
    # ------------------------------------------------------------------
    def _build_messages(self, prompt: str, image: Image.Image):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _apply_template(self, messages, add_generation_prompt=False):
        kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
        try:
            return self.processor.apply_chat_template(
                messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.processor.apply_chat_template(messages, **kwargs)

    # ------------------------------------------------------------------
    # Tokenise one set of texts + images (single processor call)
    # ------------------------------------------------------------------
    def _tokenize_batch(
        self,
        texts: List[str],
        images: List[List[Image.Image]],
    ) -> Dict[str, torch.Tensor]:
        return self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

    # ------------------------------------------------------------------
    # dtype cast (match Unsloth reference)
    # ------------------------------------------------------------------
    def _cast_pixel_values(self, batch: Dict) -> Dict:
        for key in ("pixel_values", "pixel_values_videos"):
            if key not in batch:
                continue
            pv = batch[key]
            if isinstance(pv, torch.Tensor) and pv.dtype != self.model_dtype:
                batch[key] = pv.to(self.model_dtype)
            elif isinstance(pv, list):
                for i, v in enumerate(pv):
                    if isinstance(v, torch.Tensor) and v.dtype != self.model_dtype:
                        pv[i] = v.to(self.model_dtype)
        return batch

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        prompt_texts: List[str] = []
        chosen_texts: List[str] = []
        rejected_texts: List[str] = []
        images: List[List[Image.Image]] = []      # list-of-lists for processor

        for example in examples:
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]
            image_path = example.get("image", "")

            image = (
                self._load_image(image_path)
                if image_path
                else Image.new("RGB", (224, 224), (128, 128, 128))
            )
            images.append([image])                 # ← list-of-lists

            prompt_msgs = self._build_messages(prompt, image)
            chosen_msgs = [{"role": "assistant", "content": [{"type": "text", "text": chosen}]}]
            rejected_msgs = [{"role": "assistant", "content": [{"type": "text", "text": rejected}]}]

            prompt_texts.append(self._apply_template(prompt_msgs, add_generation_prompt=True))
            chosen_texts.append(self._apply_template(prompt_msgs + chosen_msgs))
            rejected_texts.append(self._apply_template(prompt_msgs + rejected_msgs))

        # --- tokenise (image processed only once per set) -----------------
        prompt_batch = self._tokenize_batch(prompt_texts, images)
        chosen_batch = self._tokenize_batch(chosen_texts, images)
        rejected_batch = self._tokenize_batch(rejected_texts, images)

        # --- labels: mask padding + prompt tokens -------------------------
        chosen_labels = chosen_batch["input_ids"].clone()
        rejected_labels = rejected_batch["input_ids"].clone()

        chosen_labels[chosen_labels == self.pad_token_id] = self.ignore_index
        rejected_labels[rejected_labels == self.pad_token_id] = self.ignore_index

        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        for i in range(len(examples)):
            pl = int(prompt_lengths[i].item())
            chosen_labels[i, :pl] = self.ignore_index
            rejected_labels[i, :pl] = self.ignore_index

        # --- assemble output batch ----------------------------------------
        batch: Dict[str, Any] = {
            "prompt_input_ids": prompt_batch["input_ids"],
            "prompt_attention_mask": prompt_batch["attention_mask"],

            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "chosen_labels": chosen_labels,

            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
            "rejected_labels": rejected_labels,
        }

        # Vision tensors (shared between chosen / rejected)
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos"):
            if key in chosen_batch:
                batch[key] = chosen_batch[key]

        # Cast pixel_values to model dtype
        batch = self._cast_pixel_values(batch)

        # mm_token_type_ids (Qwen3.5 M-RoPE): shape = [batch, seq_len]
        # Must be split per chosen/rejected so the trainer can pad + concat
        if "mm_token_type_ids" in chosen_batch:
            batch["chosen_mm_token_type_ids"] = chosen_batch["mm_token_type_ids"]
            batch["rejected_mm_token_type_ids"] = rejected_batch["mm_token_type_ids"]

        return batch
