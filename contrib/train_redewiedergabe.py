import argparse
import itertools
import logging
from datetime import datetime
from pathlib import Path

import flair
import more_itertools
import pandas
import torch
from flair.data import Sentence, Token, Corpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


def token_iterator(df_doc):
    for _, row in df_doc.iterrows():
        if row['tok'] == 'EOF':
            continue
        yield row


def gen_sentences(toks, max_size=300):
    for sent in more_itertools.split_before(toks, lambda x: x['sentstart'] == 'yes'):
        yield sent[:max_size]


def gen_inputs(df_doc, max_size=300):
    sentences = gen_sentences(token_iterator(df_doc))
    for list_of_sentences in more_itertools.constrained_batches(sentences, max_size=max_size):
        yield itertools.chain(*list_of_sentences)


def tsv_to_inputs(path, kind):
    df = pandas.read_csv(path, sep='\t')
    for _, df_doc in df.groupby('file'):

        for input in gen_inputs(df_doc):
            flair_tokens = []
            for i in input:
                flair_tok = Token(str(i['tok']))
                if i['cat'] == kind:
                    flair_tok.add_label('cat', kind)
                else:
                    flair_tok.add_label('cat', 'O')
                flair_tokens.append(flair_tok)

            yield Sentence(flair_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rwfiles', type=Path, help='Path with the REDEWIEDERGABE train/dev/test split as TSV files.',
                        required=True)
    parser.add_argument('--output', type=Path, help='Training output', default='train_output')
    args = parser.parse_args()

    flair.device = torch.device('cuda')
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    timestamp = datetime.now().isoformat(timespec='seconds')
    for kind in ['reported', 'freeIndirect', 'direct', 'indirect']:
        logger.info(f'training {kind}')
        train_data = list(tsv_to_inputs(str(args.rwfiles / 'train' / f'{kind}_combined.tsv'), kind))
        dev_data = list(tsv_to_inputs(str(args.rwfiles / 'val' / f'{kind}_combined.tsv'), kind))
        test_data = list(tsv_to_inputs(str(args.rwfiles / 'test' / f'{kind}_combined.tsv'), kind))
        corpus = Corpus(train_data, dev_data, test_data)
        tag_dictionary = corpus.make_label_dictionary(label_type="cat", add_unk=False)

        embeddings = TransformerWordEmbeddings(
            'lkonle/fiction-gbert-large',  # which transformer model
            layers="-1",  # which layers (here: only last layer when fine-tuning)
            pooling_operation='first_last',  # how to pool over split tokens
            fine_tune=True,  # whether or not to fine-tune
        )

        outdir = args.output / "redewiedergabe" / timestamp / kind
        outdir.mkdir(parents=True)
        tagger = SequenceTagger(hidden_size=256,
                                embeddings=embeddings,
                                tag_dictionary=tag_dictionary,
                                tag_type="cat",
                                use_crf=False,
                                use_rnn=False,
                                reproject_embeddings=False,
                                )

        trainer = ModelTrainer(tagger, corpus)

        # fine-tune with setting from BERT paper
        trainer.train(str(outdir),
                      learning_rate=5e-6,  # very low learning rate
                      optimizer=torch.optim.AdamW,
                      mini_batch_size=4,
                      mini_batch_chunk_size=2,  # set this if you get OOM errors
                      max_epochs=20,  # very few epochs of fine-tuning
                      )
