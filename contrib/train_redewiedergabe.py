import argparse
import itertools
from datetime import datetime
from pathlib import Path
import numpy as np
import re
from peft import LoraConfig, get_peft_model


import pandas
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report, f1_score
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, \
    TrainingArguments, Trainer


ALL_STWR = list(itertools.product(['direct', 'indirect', 'freeIndirect', 'reported'], ['speech', 'thought', 'writing']))
id2label = {0: 'O', 1: 'B', 2: 'I'}
label2id = {v: k for k, v in id2label.items()}


def read_speech_label(speech_label):
    m = re.match(r'([^.]+)\.([^.]+)\.\d+(\.(.+))?$', speech_label)
    if m == None:
        print(speech_label)
        raise AssertionError()
    if m.group(4) != None:
        if any(m.group(4).startswith(y) for y in {'nonfact', 'border', 'prag', 'metaph'}):
            return None
        else:
            print(m.group(3))
            raise AssertionError()

    medium = m.group(2)
    speech_type = m.group(1)

    if medium == 'speech thought':
        medium = 'thought'
    if medium == 'speech writing':
        medium = 'writing'
    if medium == 'thought writing':
        medium = 'writing'
    if medium == 'speech thought writing':
        return None

    if speech_type == 'indirect freeIndirect':
        speech_type = 'freeIndirect'

    assert ' ' not in medium and ' ' not in speech_type

    return medium, speech_type


def load_file(tsv_path):
    df = pandas.read_csv(tsv_path, sep='\t', na_values=[''], keep_default_na=False, quoting=3)
    
    speech_labels = set(y for x in df['stwr'].dropna() for y in x.split('|') if y != '-')
    document_speeches = []
    for speech_label in speech_labels:
        # speech_id = int(re.search(r'\d+', speech_label).group())
        speech_idx = df[df['stwr'].apply(lambda x: speech_label in x.split('|'))].index.values

        res = read_speech_label(speech_label)
        if res is None:
            continue
        else:
            medium, speech_type = res
        document_speeches.append([speech_idx.tolist(), medium, speech_type])

    output = pandas.DataFrame(index=df.index)
    output['token'] = df['tok']
    for speech_type, medium in ALL_STWR:
        output[f'{speech_type}_{medium}_tags'] = 'O'

    for speech_idx, medium, speech_type in document_speeches:
        col = output[f'{speech_type}_{medium}_tags']
        begin = speech_idx[0]
        if begin in col.index:
            col.loc[begin] = 'B'
        for inside in speech_idx[1:]:
            if inside in col.index:
                col.loc[inside] = 'I'

    yield output.to_dict(orient='list')

def build_dataset(rw_dir):
    all_files = [x for x in rw_dir.glob('*.tsv') if 'metadata' not in x.name]
    lit_files = [x for x in all_files if 'digbib' in x.name]

    rng = np.random.default_rng(seed=123)
    test_files = set(rng.choice(list(sorted(lit_files)), size=int(0.2 * len(lit_files)), replace=False))
    train_files = set(all_files) - test_files

    def gen_examples(files):
        for t in files:
            yield from load_file(t)

    train_ds = Dataset.from_generator(gen_examples, gen_kwargs=dict(files=train_files))
    test_ds = Dataset.from_generator(gen_examples, gen_kwargs=dict(files=test_files))

    return DatasetDict(dict(train=train_ds, test=test_ds))

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else label2id[labels[word_id]]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            new_labels.append(-100)

    return new_labels

def tokenize_and_align_labels(examples, tokenizer, max_length=None):
    tokenized_inputs = tokenizer(
        examples['token'], is_split_into_words=True, max_length=max_length, return_overflowing_tokens=True
    )

    new_labels = []
    for batch_index in range(len(tokenized_inputs['input_ids'])):
        new_batch_labels = []
        sample_id = tokenized_inputs['overflow_to_sample_mapping'][batch_index]
        batch_word_ids = tokenized_inputs.word_ids(batch_index)
        for speech_type, medium in ALL_STWR:
            field = f'{speech_type}_{medium}_tags'
            new_stwr_labels = align_labels_with_tokens(examples[field][sample_id], batch_word_ids)
            new_batch_labels.append(new_stwr_labels)

        new_labels.append(np.array(new_batch_labels).T)

    tokenized_inputs.pop('overflow_to_sample_mapping')
    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs


def calc_weight(tokenized_train):
    label_df = pandas.DataFrame(np.concatenate(tokenized_train['labels']), columns=ALL_STWR)
    freq = label_df[label_df[('direct', 'speech')] != -100].apply(lambda x: x.value_counts())
    weight = 1 / freq.values
    return torch.tensor(weight).to('cuda').float()

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    batch_size, seq_len = preds.shape[:2]
    preds = preds.reshape(batch_size, seq_len, len(ALL_STWR), 3).argmax(axis=3)

    def make_onehot(speech_type, medium, bio_tag_id):
        i = ALL_STWR.index((speech_type, medium)) * 3 + bio_tag_id
        return np.eye(len(ALL_STWR) * 3, dtype=int)[i]

    y_true, y_pred = [], []
    for i, (speech_type, medium) in enumerate(ALL_STWR):
        for pred, label in zip(preds[:,:,i], labels[:,:,i]):
            decoded_pred = [p for (p, l) in zip(pred, label) if l != -100]
            decoded_label = [l for (p, l) in zip(pred, label) if l != -100]

            for pred_, label_ in zip(decoded_pred, decoded_label):
                y_true.append(make_onehot(speech_type, medium, label_))
                y_pred.append(make_onehot(speech_type, medium, pred_))

    y_true = np.stack(y_true)
    y_pred = np.stack(y_pred)
    target_names = np.array(['.'.join(x)+'.'+id2label[id_] for x in ALL_STWR for id_ in range(3)])
    label_ids = [i for i in range(len(target_names)) if not target_names[i].endswith('O')]
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names[label_ids], labels=label_ids,
                                   zero_division=0)
    print(report)

    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro', labels=label_ids, zero_division=0)
    return {'f1_score': f1}

def compute_metrics_simplified(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    batch_size, seq_len = preds.shape[:2]
    preds = preds.reshape(batch_size, seq_len, len(ALL_STWR), 3).argmax(axis=3)

    def make_onehot(speech_type, medium, bio_tag_id):
        i = ALL_STWR.index((speech_type, medium))
        return np.eye(len(ALL_STWR), dtype=int)[i] * (bio_tag_id != label2id['O'])

    y_true, y_pred = [], []
    for i, (speech_type, medium) in enumerate(ALL_STWR):
        for pred, label in zip(preds[:,:,i], labels[:,:,i]):
            decoded_pred = [p for (p, l) in zip(pred, label) if l != -100]
            decoded_label = [l for (p, l) in zip(pred, label) if l != -100]

            for pred_, label_ in zip(decoded_pred, decoded_label):
                y_true.append(make_onehot(speech_type, medium, label_))
                y_pred.append(make_onehot(speech_type, medium, pred_))

    y_true = np.stack(y_true)
    y_pred = np.stack(y_pred)
    target_names = np.array(['.'.join(x) for x in ALL_STWR])
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names, zero_division=0)
    print(report)

    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    return {'f1_score': f1}



def calc_loss(outputs, labels, weights):
    logits = outputs.logits
    batch_size, seq_len = labels.shape[:2]
    logits = logits.reshape(batch_size, seq_len, len(ALL_STWR), 3).permute(0, 3, 1, 2)
    # logits.shape == (batch_size, 3, seq_len, stwr)
    # labels.shape == (batch_size, seq_len, stwr)
    loss = CrossEntropyLoss(reduction='none')(logits, labels)
    batch_weight = torch.zeros_like(loss).float()
    for b in range(batch_size):
        for i in range(len(ALL_STWR)):
            mask = labels[b,:,i] != -100
            batch_weight[b,mask,i] = weights[labels[b,mask,i],i] #1 #weight[i,labels[mask,i]]
    loss = (loss*batch_weight).sum() / batch_weight.sum()

    return loss


#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rwfiles', type=Path,
                        help='Path with the REDEWIEDERGABE main TSV files, usually directory `data/main/tsv` '
                             'in the REDEWIEDERGABE repository.',
                        required=True)
    parser.add_argument('--output', type=Path, help='Training output', default='train_output')
    args = parser.parse_args()

    timestamp = datetime.now().isoformat(timespec='seconds')
    outdir = args.output / "redewiedergabe" / Path(timestamp)
    outdir.mkdir(parents=True)

    # rwfiles = Path('/home/ehrmanntraut/mnt/corpora/corpus_redewiedergabe/data/main/tsv')
    ds = build_dataset(args.rwfiles)

    model_id = 'LSX-UniWue/ModernGBERT_1B'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenized_datasets = ds.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs=dict(tokenizer=tokenizer),
        remove_columns=ds["train"].column_names,
    )

    weights = calc_weight(tokenized_datasets['train'])
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=[-100] * len(ALL_STWR))
    model_id2label = dict(enumerate(f"{speech_type}.{medium}.{bio}" for (speech_type, medium), bio in itertools.product(ALL_STWR, 'OBI')))

    model = AutoModelForTokenClassification.from_pretrained(model_id, num_labels=3 * len(ALL_STWR),
                                                            id2label=model_id2label, label2id={v: k for k, v in model_id2label.items()})
    peft_config = LoraConfig(
        task_type="TOKEN_CLS", r=8, lora_alpha=32,
        target_modules=["Wqkv", "Wi", "Wo"]
    )
    model = get_peft_model(model, peft_config)

    def my_loss(outputs, labels, **kwargs):
        return calc_loss(outputs, labels, weights)

    def my_metrics(eval_preds):
        ret = compute_metrics(eval_preds)
        compute_metrics_simplified(eval_preds)
        return ret

    training_args = TrainingArguments(
        output_dir=str(outdir),
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=20,
        warmup_steps=5,
        weight_decay=0.01,
        log_level='info',
        eval_strategy="epoch",
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        metric_for_best_model='eval_f1_score',
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=my_metrics,
        compute_loss_func=my_loss,
    )

    trainer.train()
    trainer.save_model(str(outdir))
    trainer.create_model_card()
