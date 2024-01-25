from __future__ import annotations
import collections
import logging
import time
from abc import abstractmethod

import regex as re
import pandas
from spacy.tokens import Doc, Token
from typing import Callable, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


class Module:

    def __init__(self, name, pbar_opts=None):
        self.name = name

        if pbar_opts is None:
            pbar_opts = {'unit': 'tok', 'postfix': self.name, 'ncols': 80, 'leave': False}
        self.pbar_opts = pbar_opts

    @abstractmethod
    def process(self, doc: Doc, pbar: tqdm) -> Doc:
            raise NotImplementedError()

    def __call__(self, doc: Doc, silent: bool = False) -> Doc:
        pbar_opts = dict(self.pbar_opts)
        if silent:
            pbar_opts['disable'] = True
        else:
            pbar_opts.update({'total': len(doc)})
        pbar = tqdm(**pbar_opts)

        start_time = time.monotonic()
        self.before_run()
        self.process(doc, pbar)
        self.after_run()
        end_time = time.monotonic()

        if not silent:
            filename = getattr(doc._, 'filename', 'stdin')
            logger.info(f'Finished module {self.name} for {filename} in {end_time-start_time:.0f}s '
                        f'({len(doc) / (end_time-start_time):.0f}tok/s)')

        pbar.close()

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
        for column in ["i", "text", "_.orig", "_.orig_offset", "is_sent_start", "_.is_para_start", "_.is_section_start", "pos_", "tag_", "lemma_", "morph", "dep_", "head"]:
            val = get_value(tok, column)
            if val is None:
                val = '_'
            if type(val) is bool:
                val = int(val)
            if type(val) not in {int, float}:
                val = str(val)
            token_attribute_dictionary[column].append(val)

    if Token.has_extension('speech'):
        for tok in doc:
            if len(tok._.speech) > 0:
                token_attribute_dictionary['speech'].append(','.join(tok._.speech))
            else:
                token_attribute_dictionary['speech'].append('_')

    for tok in doc:
        ent_str = 'O' if tok.ent_iob_ == 'O' else tok.ent_iob_ + '-' + tok.ent_type_
        token_attribute_dictionary['entity'].append(ent_str)

    if hasattr(doc._, 'characters'):
        for tok in doc:
            char_str = 'O' if tok._.character_iob == 'O' else tok._.character_iob + '-PER'
            token_attribute_dictionary['character'].append(char_str)

    if hasattr(doc._, 'coref_clusters'):
        for tok in doc:
            if len(tok._.coref_clusters) > 0:
                token_attribute_dictionary['coref_clusters'].append(
                    ','.join([str(cluster.attrs['id']) for cluster in tok._.coref_clusters]))
            else:
                token_attribute_dictionary['coref_clusters'].append('_')

    if Token.has_extension('emotions'):
        for tok in doc:
            if len(tok._.emotions) > 0:
                token_attribute_dictionary['emotions'].append(','.join(f'{span.label_}-{span.id}' for span in tok._.emotions))
            else:
                token_attribute_dictionary['emotions'].append('_')

    if Token.has_extension('scene'):
        for tok in doc:
            token_attribute_dictionary['scene_id'] = int(tok._.scene.id)
            token_attribute_dictionary['scene_label'] = tok._.scene.label_

    df = pandas.DataFrame(token_attribute_dictionary).set_index('i')

    if hasattr(doc._, 'events'):
        df['event_id'] = '_'
        df['event_label'] = '_'
        for i, event in enumerate(doc._.events):
            for span in event:
                token_ids = list(sorted(tok.i for tok in span))
                df.loc[token_ids, 'event_id'] = int(i)
                df.loc[token_ids, 'event_label'] = event.attrs['event_type']

    df = df.rename(columns=lambda x: re.sub(r'(_$|^_\.)', '', x))
    df = df.fillna(value='_').replace('', '_')
    return df
