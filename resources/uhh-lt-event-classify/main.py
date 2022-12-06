import logging
import math
import os
from collections import Counter
from typing import Iterator, List, Optional

import catma_gitlab as catma
import hydra
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import ElectraForSequenceClassification, ElectraTokenizer

import event_classify.datasets
from event_classify.config import Config, DatasetConfig, DatasetKind
from event_classify.datasets import (
    SimpleEventDataset,
    SimpleJSONEventDataset,
    SpanAnnotation,
)
from event_classify.eval import evaluate
from event_classify.label_smoothing import LabelSmoothingLoss
from event_classify.model import ElectraForEventClassification


def add_special_tokens(model, tokenizer):
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": ["<ee>", "<se>"],
        }
    )
    model.resize_token_embeddings(len(tokenizer))


def train(train_loader, dev_loader, model, config: Config):
    model.to(config.device)
    optimizer = SGD(model.parameters(), lr=config.learning_rate)
    f1s: List[float] = []
    scheduler: Optional[LambdaLR] = None
    if config.scheduler.enable:
        scheduler = LambdaLR(
            optimizer,
            lambda epoch: 1 - (epoch / config.scheduler.epochs),
        )
    for epoch in range(config.epochs):
        loss_epoch: float = 0.0
        losses = []
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, (input_data, labels, _) in enumerate(pbar):
            out = model(**input_data.to(config.device), labels=labels.to(config.device))
            loss = out.loss
            loss_epoch += float(loss.item())
            pbar.set_postfix({"mean epoch loss": loss_epoch / (i + 1)})
            losses.append(loss.item())
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            model.zero_grad()
            if i % config.loss_report_frequency == 0:
                mlflow.log_metric("Loss", sum(losses) / len(losses), i)
            while len(losses) > config.loss_report_frequency:
                losses.pop(0)
            if i % config.loss_report_frequency == 0 and config.dynamic_loss_weighting:
                for x in range(len(config.optimize_outputs)):
                    mlflow.log_metric(
                        f"Sigma_{x + 1}", model.multi_loss.log_sigmas[x].item()
                    )
        if scheduler is not None:
            mlflow.log_metric("Learning Rate", scheduler.get_last_lr()[0], epoch)
            scheduler.step()
        if dev_loader is not None:
            evaluation_results = evaluate(dev_loader, model, config.device, epoch=epoch)
            print("Logging metrics: ", evaluation_results.extra_metrics)
            mlflow.log_metrics(evaluation_results.extra_metrics)
            assert evaluation_results.macro_f1 is not None
            assert evaluation_results.weighted_f1 is not None
            if config.optimize == "weighted f1":
                f1: float = float(evaluation_results.weighted_f1)
            elif config.optimize == "macro f1":
                f1: float = float(evaluation_results.macro_f1)
            else:
                logging.warning("Invalid optimization metric, defaulting to weighted.")
                f1 = evaluation_results.weighted_f1
            if (len(f1s) > 0 and f1 > max(f1s)) or len(f1s) == 0:
                model.save_pretrained("best-model")
            f1s.append(f1)
            mlflow.log_metric("Weighted F1", evaluation_results.weighted_f1, epoch)
            mlflow.log_metric("Macro F1", evaluation_results.macro_f1, epoch)
            if len(f1s) > 0 and max(f1s) not in f1s[-config.patience :]:
                logging.info("Ran out of patience, stopping training.")
                return


def get_datasets(config: DatasetConfig) -> tuple[Dataset]:
    if config.kind == DatasetKind.CATMA:
        if config.catma_dir is None or config.catma_uuid is None:
            raise ValueError(
                "When chosing catma dataset kind, you must specify a catma_directory and catma_uuid!"
            )
        project = catma.CatmaProject(
            hydra.utils.to_absolute_path(config.catma_dir),
            config.catma_uuid,
            filter_intrinsic_markup=False,
        )
    if config.in_distribution:
        included_collections, _ = event_classify.datasets.get_annotation_collections(
            config.excluded_collections,
        )
        if config.kind == DatasetKind.CATMA.value:
            dataset = SimpleEventDataset(
                project,
                included_collections,
                include_special_tokens=config.special_tokens,
            )
        elif config.kind == DatasetKind.JSON.value:
            dataset = SimpleJSONEventDataset(
                os.path.join(
                    hydra.utils.get_original_cwd(), "data/forTEXT-EvENT_Dataset-e6bc150"
                ),
                include_special_tokens=config.special_tokens,
            )
        else:
            raise ValueError("Invalid dataset kind!")
        total = len(dataset)
        train_size = math.floor(total * 0.8)
        dev_size = (total - train_size) // 2
        test_size = total - train_size - dev_size
        train_dataset, dev_dataset, test_dataset = random_split(
            dataset,
            [train_size, dev_size, test_size],
            generator=torch.Generator().manual_seed(13),
        )
    else:
        (
            included_collections,
            ood_collections,
        ) = event_classify.datasets.get_annotation_collections(
            config.excluded_collections,
        )
        in_distribution_dataset = SimpleEventDataset(
            project,
            included_collections,
            include_special_tokens=config.special_tokens,
        )
        train_size = math.floor(len(in_distribution_dataset) * 0.9)
        dev_size = len(in_distribution_dataset) - train_size
        train_dataset, dev_dataset = random_split(
            in_distribution_dataset,
            [train_size, dev_size],
            generator=torch.Generator().manual_seed(13),
        )
        test_dataset = SimpleEventDataset(
            project,
            ood_collections,
            include_special_tokens=config.special_tokens,
        )
    return train_dataset, dev_dataset, test_dataset


def build_loaders(
    tokenizer: ElectraTokenizer, datasets: List[Dataset], config: Config
) -> Iterator[DataLoader]:
    for ds in datasets:
        if ds:
            yield DataLoader(
                ds,
                batch_size=config.batch_size,
                collate_fn=lambda list_: SpanAnnotation.to_batch(list_, tokenizer),
                shuffle=True,
            )
        else:
            yield None


def print_target_weights(dataset):
    counts = Counter(el.event_type for el in dataset)
    logging.info("Recommended class weights:")
    output_classes = []
    output_weights = []
    for event_type, value in sorted(counts.items(), key=lambda x: x[0].value):
        weight = 1 / value
        logging.info(f"Class: {event_type}, {weight}")


@hydra.main(config_name="conf/config")
def main(config: Config):
    hydra_run_name = HydraConfig.get().run.dir.replace("outputs/", "").replace("/", "_")
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")
    with mlflow.start_run(run_name=hydra_run_name):
        return _main(config)


def _main(config: Config):
    tokenizer: ElectraTokenizer = ElectraTokenizer.from_pretrained(
        config.pretrained_model,
    )
    tokenizer.save_pretrained("tokenizer")
    model = ElectraForEventClassification.from_pretrained(
        config.pretrained_model,
        event_config=config,
    )
    mlflow.log_params(dict(config))
    add_special_tokens(model, tokenizer)
    datasets = get_datasets(config.dataset)
    print_target_weights(datasets[0])
    assert datasets[0] is not None
    assert datasets[-1] is not None
    train_loader, dev_loader, test_loader = list(
        build_loaders(tokenizer, datasets, config)
    )
    train(train_loader, dev_loader, model, config)
    if dev_loader is not None:
        model = ElectraForEventClassification.from_pretrained(
            "best-model",
            event_config=config,
        )
    logging.info("Dev set results")
    evaluate(
        dev_loader,
        model,
        device=config.device,
        out_file=open("predictions-dev.json", "w"),
    )
    logging.info("Test set results")
    results = evaluate(
        test_loader,
        model,
        device=config.device,
        out_file=open("predictions.json", "w"),
        save_confusion_matrix=True,
    )
    return results.weighted_f1


if __name__ == "__main__":
    main()
