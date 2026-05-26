import json

import pytest
import torch
from PIL import Image

from open_clip_train.data import RetrievalDataset
from open_clip_train.retrieval import _compute_retrieval_metrics


def test_retrieval_metrics_support_multiple_captions_per_image():
    logits = torch.tensor([
        [0.9, 0.8, 0.7, 0.2, 0.1, 0.0],
        [0.1, 0.0, 0.2, 0.9, 0.8, 0.7],
    ])
    image_ids = ["image-0", "image-1"]
    text_ids = [0, 1, 2, 3, 4, 5]
    image_to_texts = {
        "image-0": [0, 1, 2],
        "image-1": [3, 4, 5],
    }
    text_to_image = {
        0: "image-0",
        1: "image-0",
        2: "image-0",
        3: "image-1",
        4: "image-1",
        5: "image-1",
    }

    metrics = _compute_retrieval_metrics(
        logits=logits,
        image_ids=image_ids,
        text_ids=text_ids,
        image_to_texts=image_to_texts,
        text_to_image=text_to_image,
    )

    assert metrics["image_to_text_R@1"] == pytest.approx(1.0)
    assert metrics["text_to_image_R@1"] == pytest.approx(1.0)


def test_retrieval_dataset_loads_karpathy_json(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 8)).save(image_root / "sample.jpg")
    annotations = {
        "images": [
            {
                "filename": "sample.jpg",
                "split": "test",
                "imgid": 42,
                "sentences": [
                    {"raw": "a first caption"},
                    {"tokens": ["a", "second", "caption"]},
                ],
            },
            {
                "filename": "ignored.jpg",
                "split": "train",
                "imgid": 43,
                "sentences": [{"raw": "not used"}],
            },
        ]
    }
    annotation_file = tmp_path / "karpathy.json"
    annotation_file.write_text(json.dumps(annotations))

    dataset = RetrievalDataset(
        image_root=str(image_root),
        annotation_file=str(annotation_file),
        transform=lambda image: image,
        tokenizer=lambda texts: texts,
        split="test",
    )

    assert len(dataset) == 2
    assert dataset.image_to_texts == {"42": [0, 1]}
    assert dataset.text_to_image == {0: "42", 1: "42"}


def test_retrieval_dataset_loads_coco_captions_json(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 8)).save(image_root / "coco.jpg")
    annotations = {
        "images": [{"id": 7, "file_name": "coco.jpg"}],
        "annotations": [
            {"image_id": 7, "id": 101, "caption": "caption one"},
            {"image_id": 7, "id": 102, "caption": "caption two"},
        ],
    }
    annotation_file = tmp_path / "captions.json"
    annotation_file.write_text(json.dumps(annotations))

    dataset = RetrievalDataset(
        image_root=str(image_root),
        annotation_file=str(annotation_file),
        transform=lambda image: image,
        tokenizer=lambda texts: texts,
    )

    assert len(dataset) == 2
    assert dataset.image_to_texts == {"7": [0, 1]}
    assert dataset.text_to_image == {0: "7", 1: "7"}
