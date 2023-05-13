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
    def process(self, doc: Doc, progress_fn: Callable[[int], None]) -> Doc:
        raise NotImplementedError()

    def __call__(self, doc: Doc, disable_pbar: bool = False, before_run_hook: Optional[Callable[[Module, Doc], None]] = None,
                 after_run_hook: Optional[Callable[[Module, Doc], None]] = None,
                 update_hook: Optional[Callable[[Module, Doc, int], None]] = None) -> Doc:
        pbar_opts = dict(self.pbar_opts)
        if disable_pbar:
            pbar_opts['disable'] = True

        pbar_opts.update({'total': len(doc)})
        pbar = tqdm(**pbar_opts)
        state_obj = {}

        def my_update_fn(x: int):
            pbar.update(x)
            if update_hook is not None:
                update_hook(self, doc, x)

        def my_before_run_hook(module, doc):
            state_obj['start_time'] = time.monotonic()

        def my_after_run_hook(module, doc):
            start_time = state_obj['start_time']
            end_time = time.monotonic()

            filename = getattr(doc._, 'filename', 'stdin')
            logger.info(f'Finished module {self.name} for {filename} in {end_time-start_time:.0f}s '
                         f'({len(doc) / (end_time-start_time):.0f}tok/s)')

        if before_run_hook is None and after_run_hook is None:
            before_run_hook = my_before_run_hook
            after_run_hook = my_after_run_hook

        before_run_hook(self, doc)
        self.before_run()
        self.process(doc, my_update_fn)
        self.after_run()
        after_run_hook(self, doc)
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
        for column in ["i", "text", "_.orig", "is_sent_start", "_.is_para_start", "_.is_section_start", "pos_", "tag_", "lemma_", "morph", "dep_", "head"]:
            val = get_value(tok, column)
            if val is None:
                val = '_'
            if type(val) is bool:
                val = int(val)
            if type(val) not in {int, float}:
                val = str(val)
            token_attribute_dictionary[column].append(val)

    if hasattr(doc._, 'speech'):
        for tok in doc:
            if len(tok._.speech) > 0:
                token_attribute_dictionary['speech'].append(','.join(tok._.speech))
            else:
                token_attribute_dictionary['speech'].append('_')

    for tok in doc:
        ent_str = 'O' if tok.ent_iob_ == 'O' else tok.ent_iob_ + '-' + tok.ent_type_
        token_attribute_dictionary['entity'].append(ent_str)
        char_str = 'O' if tok._.character_iob == 'O' else tok._.character_iob + '-PER'
        token_attribute_dictionary['character'].append(char_str)

    if hasattr(doc._, 'coref_clusters'):
        for tok in doc:
            if len(tok._.coref_clusters) > 0:
                token_attribute_dictionary['coref_clusters'].append(
                    ','.join([str(cluster.attrs['id']) for cluster in tok._.coref_clusters]))
            else:
                token_attribute_dictionary['coref_clusters'].append('_')

    df = pandas.DataFrame(token_attribute_dictionary).set_index('i')

    if hasattr(doc._, 'scenes'):
        df['scene_id'] = '_'
        df['scene_label'] = '_'
        for scene in doc._.scenes:
            for tok in scene:
                df.loc[tok.i, 'scene_id'] = int(scene.id)
                df.loc[tok.i, 'scene_label'] = scene.label_

    if hasattr(doc._, 'events'):
        df['event_id'] = '_'
        df['event_label'] = '_'
        for i, event in enumerate(doc._.events):
            for span in event:
                for tok in span:
                    df.loc[tok.i, 'event_id'] = int(i)
                    df.loc[tok.i, 'event_label'] = event.attrs['event_type']

    df = df.rename(columns=lambda x: re.sub(r'(_$|^_\.)', '', x))
    df = df.fillna(value='_').replace('', '_')
    return df
