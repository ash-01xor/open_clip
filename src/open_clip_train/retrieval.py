import logging

import torch
from tqdm import tqdm

from open_clip import get_input_dtype
from open_clip_train.precision import get_autocast


RETRIEVAL_DATASETS = {
    "flickr-val": "flickr",
    "mscoco-val": "mscoco",
}


def _as_list(values):
    if torch.is_tensor(values):
        return values.detach().cpu().tolist()
    return list(values)


def _recall_at_k(ranks, ks=(1, 5, 10)):
    if not ranks:
        return {f"R@{k}": 0.0 for k in ks}
    ranks = torch.tensor(ranks, dtype=torch.long)
    return {f"R@{k}": (ranks < k).float().mean().item() for k in ks}


def _compute_retrieval_metrics(logits, image_ids, text_ids, image_to_texts, text_to_image):
    text_id_to_col = {text_id: col for col, text_id in enumerate(text_ids)}
    image_id_to_row = {image_id: row for row, image_id in enumerate(image_ids)}

    image_ranks = []
    image_ranking = torch.argsort(logits, dim=1, descending=True)
    image_positions = torch.empty_like(image_ranking)
    image_positions.scatter_(1, image_ranking, torch.arange(logits.shape[1]).expand_as(image_ranking))
    for row, image_id in enumerate(image_ids):
        positive_cols = {
            text_id_to_col[text_id]
            for text_id in image_to_texts[image_id]
            if text_id in text_id_to_col
        }
        if not positive_cols:
            continue
        image_ranks.append(min(image_positions[row, col].item() for col in positive_cols))

    text_ranks = []
    text_ranking = torch.argsort(logits.t(), dim=1, descending=True)
    text_positions = torch.empty_like(text_ranking)
    text_positions.scatter_(1, text_ranking, torch.arange(logits.shape[0]).expand_as(text_ranking))
    for row, text_id in enumerate(text_ids):
        image_id = text_to_image[text_id]
        if image_id not in image_id_to_row:
            continue
        positive_row = image_id_to_row[image_id]
        text_ranks.append(text_positions[row, positive_row].item())

    metrics = {}
    for name, ranks in (("image_to_text", image_ranks), ("text_to_image", text_ranks)):
        recalls = _recall_at_k(ranks)
        metrics.update({f"{name}_{metric}": value for metric, value in recalls.items()})
    return metrics


def run_retrieval(model, dataloader, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    image_features_by_id = {}
    text_features_by_id = {}

    with torch.inference_mode():
        for images, texts, image_ids, text_ids in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            image_ids = [str(image_id) for image_id in _as_list(image_ids)]
            text_ids = _as_list(text_ids)

            with autocast():
                model_out = model(images, texts)
                image_features = model_out["image_features"]
                text_features = model_out["text_features"]
                logit_scale = model_out["logit_scale"].mean()

            for image_id, image_feature in zip(image_ids, image_features):
                if image_id not in image_features_by_id:
                    image_features_by_id[image_id] = image_feature.detach().cpu()
            for text_id, text_feature in zip(text_ids, text_features):
                text_features_by_id[text_id] = text_feature.detach().cpu()

    image_ids = list(image_features_by_id)
    text_ids = list(text_features_by_id)
    image_features = torch.stack([image_features_by_id[image_id] for image_id in image_ids])
    text_features = torch.stack([text_features_by_id[text_id] for text_id in text_ids])
    logits = logit_scale.detach().cpu() * image_features @ text_features.t()

    dataset = dataloader.dataset
    return _compute_retrieval_metrics(
        logits=logits,
        image_ids=image_ids,
        text_ids=text_ids,
        image_to_texts=dataset.image_to_texts,
        text_to_image=dataset.text_to_image,
    )


def retrieval_eval(model, data, epoch, args):
    eval_datasets = {
        data_key: metric_prefix
        for data_key, metric_prefix in RETRIEVAL_DATASETS.items()
        if data_key in data
    }
    if not eval_datasets:
        return {}
    if not args.val_frequency:
        return {}
    if (epoch % args.val_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info("Starting retrieval evaluation.")
    results = {}
    for data_key, metric_prefix in eval_datasets.items():
        metrics = run_retrieval(model, data[data_key].dataloader, args)
        results.update({f"{metric_prefix}-{name}": value for name, value in metrics.items()})
    logging.info("Finished retrieval evaluation.")
    return results
