import argparse
import itertools
import logging
import re
from datetime import datetime
from pathlib import Path

import flair
import more_itertools
import pandas
import torch
from bs4 import BeautifulSoup
from flair.data import Token, Sentence, Corpus, get_spans_from_bio
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


def parse_tei(tei):
    body = tei.find("body")
    paragraphs = body.find_all("p")
    output = []
    paragraphnum = 0
    myfname = tei.find('titlestmt').find('title').text

    for p in paragraphs:
        words = p.find_all("w")
        for word in words:
            parent = word.parent
            wid = word.attrs["xml:id"]

            wordId = int(re.sub("w", "", wid))
            if parent.name == "persname":
                try:
                    nertype = parent.attrs["type"]
                    if "prev" not in parent.attrs:
                        corefID = parent.attrs["xml:id"]
                    else:
                        corefID = parent.attrs["prev"][1:]
                    output.append([myfname, paragraphnum, wordId, word.text, "-", nertype,
                                   re.sub("cr", "", corefID)])
                except:
                    output.append([myfname, paragraphnum, wordId, word.text, "-", "-", "-"])
            else:
                output.append([myfname, paragraphnum, wordId, word.text, "-", "-", "-"])
        paragraphnum += 1

    res = pandas.DataFrame(output)
    res.columns = ["source", "paragraph", "tokenID", "token", "label", "nerType", "coredID"]
    res = res.set_index(['source', 'tokenID'])

    labels = []
    oldID = ""

    for index, row in res.iterrows():
        if row["nerType"] not in ['AppA', 'AppTdfW', 'Core']:
            labels.append("-")
        else:
            if oldID == row["coredID"]:
                labels.append("I-PER")
            else:
                labels.append("B-PER")

        oldID = row["coredID"]

    res["label"] = labels

    res['sentstart'] = False
    for j in body.find_all('join'):
        if j.attrs["results"] != 's':
            continue
        sentstart = int(j.attrs["target"].split(' ')[0].replace('#w', '')) - 1
        res.loc[(myfname, sentstart), 'sentstart'] = True

    res = res.drop(columns="paragraph")
    return res


def gen_sentences(lines, max_size=300):
    for sent in more_itertools.split_before(lines, lambda x: x['sentstart']):
        yield sent[:max_size]


def gen_inputs(df, max_size=300):
    sentences = gen_sentences([row for _, row in df.iterrows()], max_size=max_size)
    for list_of_sentences in more_itertools.constrained_batches(sentences, max_size=max_size):
        yield list(itertools.chain.from_iterable(list_of_sentences))


def dataframe_to_inputs(df):
    for _, df_doc in df.groupby('source'):
        for input_seq in gen_inputs(df_doc):
            flair_tokens = []
            labels = [tok['label'] if tok['label'] != '-' else 'O' for tok in input_seq]
            for tok in input_seq:
                flair_tok = Token(str(tok["token"]))
                flair_tokens.append(flair_tok)

            sent = Sentence(flair_tokens)
            for span_indices, _, value in get_spans_from_bio(labels):
                span = sent[span_indices[0]: span_indices[-1] + 1]
                if value != "O":
                    span.add_label('character', value=value)
            yield sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--droc', type=Path, help='Path to the DROC-TEI/DROC-RELEASE.xml file', required=True)
    parser.add_argument('--output', type=Path, help='Training output', default='train_output')
    args = parser.parse_args()

    flair.device = torch.device('cuda')
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    timestamp = datetime.now().isoformat(timespec='seconds')

    soup = BeautifulSoup(open(args.droc).read(), 'html.parser')

    dfs = []
    for tei in soup.find_all('tei'):
        dfs.append(parse_tei(tei))

    droc_dataset = pandas.concat(dfs).reset_index()

    corpus_split = {'Die Bernsteinhexe': 'dev',
                    'Auf fremden Pfaden': 'dev',
                    'Die gute Schule': 'dev',
                    'Mathilde Möhring': 'dev',
                    'Uli der Pächter': 'dev',
                    'Problematische Naturen. Zweite Abtheilung (Durch Nacht zum Licht)': 'dev',
                    'Das Narrenspital': 'dev',
                    'Im Reiche des silbernen Löwen IV': 'dev',
                    'Lingam': 'dev',
                    'Einer Mutter Sieg': 'dev',
                    'Schach von Wuthenow': 'test',
                    'Agave': 'test',
                    'Madonna. Unterhaltungen mit einer Heiligen': 'test',
                    'Der Stechlin': 'test',
                    'Zweiter Band': 'test',
                    'Richard Wood': 'test',
                    'Amalie. Eine wahre Geschichte in Briefen': 'test',
                    'Die Heiteretei und ihr Widerspiel': 'test'}

    droc_dataset['split'] = droc_dataset['source'].apply(lambda x: corpus_split.get(x, 'train'))

    train_data = list(dataframe_to_inputs(droc_dataset[droc_dataset.split == 'train']))
    dev_data = list(dataframe_to_inputs(droc_dataset[droc_dataset.split == 'dev']))
    test_data = list(dataframe_to_inputs(droc_dataset[droc_dataset.split == 'test']))
    corpus = Corpus(train_data, dev_data, test_data)
    tag_dictionary = corpus.make_label_dictionary(label_type="character", add_unk=False)

    timestamp = datetime.now().isoformat(timespec='seconds')
    embeddings = TransformerWordEmbeddings(
        'lkonle/fiction-gbert-large',  # which transformer model
        layers="-1",  # which layers (here: only last layer when fine-tuning)
        pooling_operation='first_last',  # how to pool over split tokens
        fine_tune=True,  # whether or not to fine-tune
    )

    outdir = args.output / "character_recognizer" / timestamp
    outdir.mkdir(parents=True)
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type="character",
                            tag_format="BIO",
                            use_crf=False,
                            use_rnn=False,
                            reproject_embeddings=False,
                            )

    trainer = ModelTrainer(tagger, corpus)

    # fine-tune with setting from BERT paper
    trainer.train(outdir,
                  learning_rate=5e-6,  # very low learning rate
                  optimizer=torch.optim.AdamW,
                  mini_batch_size=4,
                  mini_batch_chunk_size=2,  # set this if you get OOM errors
                  max_epochs=30,  # very few epochs of fine-tuning
                  )
