import logging
import time

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
