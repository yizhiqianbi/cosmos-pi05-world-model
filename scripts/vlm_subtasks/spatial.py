from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]


def select_detection(detections: list[Detection]) -> Detection | None:
    return max(detections, key=lambda item: item.score, default=None)


def binary_mask_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a mask with uncompressed COCO-style runs in column-major order."""
    mask = np.asarray(mask, dtype=np.uint8)
    flat = mask.reshape(-1, order="F")
    counts: list[int] = []
    previous = 0
    run = 0
    for element in flat:
        value = int(bool(element))
        if value == previous:
            run += 1
        else:
            counts.append(run)
            run = 1
            previous = value
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


class GroundingDinoDetector:
    """Lazy local Grounding DINO backend using Hugging Face Transformers."""

    def __init__(self, model: str = "IDEA-Research/grounding-dino-base", *, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection
        from transformers import AutoProcessor

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model).to(device).eval()

    def detect(self, image: np.ndarray, query: str, *, threshold: float = 0.25) -> list[Detection]:
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
        inputs = self.processor(images=pil, text=query, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=[pil.size[::-1]],
        )[0]
        labels = result.get("text_labels", result.get("labels", []))
        return [
            Detection(str(label), float(score), tuple(float(value) for value in box))
            for label, score, box in zip(labels, result["scores"], result["boxes"], strict=True)
        ]


class Sam2Segmenter:
    """SAM2 image predictor used on task-relevant event frames."""

    def __init__(self, config: str, checkpoint: str, *, device: str = "cuda") -> None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.predictor = SAM2ImagePredictor(build_sam2(config, checkpoint, device=device))

    def segment(self, image: np.ndarray, box_xyxy: tuple[float, float, float, float]) -> tuple[np.ndarray, float]:
        self.predictor.set_image(np.asarray(image, dtype=np.uint8))
        masks, scores, _ = self.predictor.predict(box=np.asarray(box_xyxy), multimask_output=True)
        index = int(np.argmax(scores))
        return np.asarray(masks[index], dtype=bool), float(scores[index])


def ground_frame(
    image: np.ndarray,
    *,
    target_object: str,
    destination: str,
    detector: GroundingDinoDetector,
    segmenter: Sam2Segmenter | None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for role, query in (("target", target_object), ("destination", destination)):
        if not query:
            continue
        detection = select_detection(detector.detect(image, query))
        if detection is None:
            output[role] = {"query": query, "status": "not_found"}
            continue
        item: dict[str, Any] = {
            "query": query,
            "status": "found",
            "label": detection.label,
            "score": detection.score,
            "bbox_xyxy": list(detection.box_xyxy),
        }
        if segmenter is not None:
            mask, mask_score = segmenter.segment(image, detection.box_xyxy)
            item.update({"mask_rle": binary_mask_rle(mask), "mask_score": mask_score})
        output[role] = item
    return output
