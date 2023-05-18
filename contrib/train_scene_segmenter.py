import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import more_itertools
import numpy as np
import sklearn
from datasets import Dataset
from transformers import Trainer, TrainingArguments, BertTokenizer, DataCollatorForTokenClassification

from llpro.components.scene_segmenter import BertForSentTransitions


def read_stssfile(json_file):
    content = json.load(open(json_file, ))
    sentences, labels, indices = [], [], []
    selected = {"Scene": [], "Nonscene": []}
    scene_borders = {range(k["begin"], k["end"]): k["type"] for k in content["scenes"]}
    for sent in content["sentences"]:
        label = None
        for k, v in scene_borders.items():
            if sent["begin"] in k:
                if k not in selected[v]:
                    label = "{}-B".format(v)
                    selected[v].append(k)
                else:
                    label = v
                break
        if not label:
            continue
        sentences.append(content["text"][sent["begin"]:sent["end"]])
        indices.append((sent["begin"], sent["end"]))
        labels.append(label)
    assert len(sentences) == len(labels) == len(indices)
    return {"sentences": sentences, "labels": labels}


def tokenize_doc(doc, tokenizer, label2id):
    sentences = doc['sentences']
    labels = doc['labels']

    def truncated_sents():
        for sent in sentences:
            tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))
            yield tokenized[:300]

    i = 0
    for chunk in more_itertools.constrained_batches(truncated_sents(), max_size=500, get_len=lambda x: len(x) + 2,
                                                    max_count=20):
        in_seq = [tokenizer.cls_token_id]
        out_seq = [-100]
        for sent in chunk:
            in_seq.extend(sent + [tokenizer.sep_token_id])
            out_seq.extend([-100] * len(sent) + [label2id[labels[i]]])
            i = i + 1

        out = tokenizer(tokenizer.decode(in_seq), add_special_tokens=False)
        out['labels'] = out_seq
        # out['sep_token_id'] = tokenizer.sep_token_id
        yield out


def stss_scorer(y_true, y_pred):
    def stss_labeling(iter):
        for a, b in more_itertools.windowed(itertools.chain([None], iter), 2):
            if a == 'Scene' and b == 'Scene-B':
                yield 'SCENE-TO-SCENE'
            elif a == 'Scene' and b == 'Nonscene-B':
                yield 'SCENE-TO-NONSCENE'
            elif a == 'Nonscene' and b == 'Scene-B':
                yield 'NONSCENE-TO-SCENE'
            elif a == 'Nonscene' and b == 'Nonscene-B':
                assert False
            else:
                yield 'NOBORDER'

    y_true = list(stss_labeling(y_true))
    y_pred = list(stss_labeling(y_pred))
    print(sklearn.metrics.classification_report(y_true=y_true, y_pred=y_pred))
    return sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro',
                                    labels=['SCENE-TO-SCENE', 'SCENE-TO-NONSCENE', 'NONSCENE-TO-SCENE'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stss', type=Path, help='Path with the prepared STTS training JOSN files.', required=True)
    parser.add_argument('--output', type=Path, help='Training output', default='train_output')
    args = parser.parse_args()

    id2label = {0: 'Scene-B', 1: 'Scene', 2: 'Nonscene-B', 3: 'Nonscene'}
    label2id = {'Scene-B': 0, 'Scene': 1, 'Nonscene-B': 2, 'Nonscene': 3}

    tokenizer = BertTokenizer.from_pretrained('lkonle/fiction-gbert-large')
    model = BertForSentTransitions.from_pretrained('lkonle/fiction-gbert-large', num_labels=4, id2label=id2label,
                                                   sep_token_id=tokenizer.sep_token_id)

    dev_files = ["9783740941093.json"]
    ds_train = []
    ds_dev = []

    for json_file in args.stss.glob('*.json'):
        print(json_file)
        doc = read_stssfile(str(json_file))
        exs = list(tokenize_doc(doc, tokenizer, label2id))
        if json_file.name in dev_files:
            ds_dev.extend(exs)
        else:
            ds_train.extend(exs)

    ds_train = Dataset.from_generator(lambda: ds_train.__iter__())
    ds_dev = Dataset.from_generator(lambda: ds_dev.__iter__())
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    timestamp = datetime.now().isoformat(timespec='seconds')
    outdir = args.output / "scene_segmenter" / Path(timestamp)
    outdir.mkdir(parents=True)
    (outdir / 'best_model').mkdir(parents=True)


    def compute_metrics(p):
        predictions, labels = p
        sentences_mask = labels != -100
        labels = labels[sentences_mask]
        predictions = predictions[sentences_mask]
        predictions = np.argmax(predictions, axis=1)
        print(sklearn.metrics.classification_report(y_true=labels, y_pred=predictions))
        return {'f1_labels': sklearn.metrics.f1_score(y_true=labels, y_pred=predictions, average='micro'),
                'f1_stss': stss_scorer(model.decode_labels(labels), model.decode_labels(predictions))}


    training_args = TrainingArguments(
        output_dir=str(outdir),
        learning_rate=5e-6,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=20,
        warmup_steps=5,
        weight_decay=0.01,
        log_level='info',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        metric_for_best_model='eval_f1_stss',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(str(outdir / "best_model"))
