import collections
import logging
import time

import regex as re
import pandas
from spacy.tokens import Doc
from typing import Callable

from tqdm import tqdm


class Module:

    def __init__(self, name, pbar_opts=None):
        self.name = name

        if pbar_opts is None:
            pbar_opts = {'unit': 'tok', 'postfix': self.name, 'dynamic_ncols': True, 'leave': False}
        self.pbar_opts = pbar_opts

    def process(self, doc: Doc, progress_fn: Callable[[int], None]) -> Doc:
        raise NotImplementedError()

    def __call__(self, doc: Doc) -> Doc:
        pbar_opts = dict(self.pbar_opts)
        pbar_opts.update({'total': len(doc)})
        pbar = tqdm(**pbar_opts)

        def my_update_fn(x: int):
            pbar.update(x)

        start_time = time.time()
        self.before_run()
        self.process(doc, my_update_fn)
        self.after_run()
        pbar.close()
        end_time = time.time()

        if 'filename' in vars(doc._):
            logging.info(
                f'Finished module {self.name} for {doc._.filename} ({len(doc) / (end_time - start_time):.0f}tok/s)')
        else:
            logging.info(f'Finished module {self.name} ({len(doc) / (end_time - start_time):.0f}tok/s)')

        return doc

    def before_run(self):
        pass

    def after_run(self):
        pass


def spacy_doc_to_dataframe(doc):
    token_attribute_dictionary = collections.defaultdict(lambda: list())

    def get_value(tok, attr):
        if attr.startswith('_.'):
            return getattr(tok._, attr[2:], None)

        if attr in ["head"]:
            return getattr(tok, attr).i

        return getattr(tok, attr)

    for tok in doc:
        for column in ["i", "text", "._.is_sent_start", "_.is_para_start", "tag_", "lemma_", "dep_", "head"]:
            token_attribute_dictionary[column].append(get_value(tok, column))

    for tok in doc:
        ent_str = 'O' if tok.ent_iob_ == 'O' else tok.ent_iob_ + '-' + tok.ent_type_
        token_attribute_dictionary['entity'].append(ent_str)

    if hasattr(doc._, 'coref_clusters'):
        for tok in doc:
            token_attribute_dictionary['coref_clusters'].append(','.join([str(cluster.attrs['id']) for cluster in tok._.coref_clusters]))

    df = pandas.DataFrame(token_attribute_dictionary).set_index('i')

    if hasattr(doc._, 'scenes'):
        df['scene'] = '_'
        for scene in doc._.scenes:
            for tok in scene:
                df.loc[tok.i, 'scene'] = f'ID={scene.id}|Label={scene.label_}'

    if hasattr(doc._, 'events'):
        df['event'] = '_'
        for i, event in enumerate(doc._.events):
            label = f'ID={i}|EventType={event.attrs["event_types"]}|SpeechType={event.attrs["speech_type"]}|ThoughtRepresentation={event.attrs["thought_representation"]}'
            for span in event:
                for tok in span:
                    df.loc[tok.i, 'event'] = label

    df = df.rename(columns=lambda x: re.sub(r'(_$|^_\.)', '', x))
    return df