import logging
import multiprocessing
import sys

import more_itertools
from spacy import Language
from spacy.tokens import Span, Doc
from typing import List, Iterable


@Language.factory("parser_parzu_parallelized", requires=['token.tag'], assigns=['token.dep', 'token.head'], default_config={'parzu_home': 'resources/ParZu', 'num_processes': 1, 'tokens_per_process': 1000})
def parser_parzu_parallelized(nlp, name, parzu_home, num_processes, tokens_per_process):
    return ParzuParallelized(name=name, parzu_home=parzu_home, num_processes=num_processes, tokens_per_process=tokens_per_process)


class ParzuParallelized:

    def __init__(self, name, parzu_home='resources/ParZu', num_processes: int = 1, tokens_per_process: int = 1000):
        self.num_processes = num_processes
        self.tokens_per_process = tokens_per_process
        logging.info(f"Starting {num_processes} processes of {name}")
        self.pool = multiprocessing.Pool(processes=num_processes, initializer=ParzuParallelized._init_worker, initargs=({'parzu_home': parzu_home},))

    def close(self):
        self.pool.close()
        self.pool.join()

    def split_doc_into_chunks(self, doc: Doc) -> Iterable[Span]:
        def sentlist_len(list_of_sents):
            return sum(len(x) for x in list_of_sents)

        for sentences in more_itertools.constrained_batches(doc.sents, max_size=self.tokens_per_process, get_len=sentlist_len):
            yield doc[sentences[0].start:sentences[-1].end]


    def __call__(self, doc: Doc) -> Doc:
        m = multiprocessing.Manager()
        q = m.Queue()
        process_chunks: List[Span] = list(self.split_doc_into_chunks(doc))
        results = []
        for i, span in enumerate(process_chunks):
            chunk_start, chunk_end = (span.start, span.end)
            results.append(self.pool.apply_async(ParzuParallelized._process_worker, (doc.vocab, doc.to_bytes(), chunk_start, chunk_end, i, q)))
        # processed_chunks: List[Span] = [None] * len(process_chunks)
        # while True:
        #     kind, i, value = q.get()
        #     if kind == 'update':
        #         pass
        #         # update_fn(value)
        #     elif kind == 'result':
        #         processed_chunks[i] = value
        #         if not any(x is None for x in processed_chunks):
        #             break
        #
        # for tok, modified_tok in zip(doc, itertools.chain.from_iterable(processed_chunks)):
        #     # TODO
        #     print(tok, modified_tok)
        #     # tok.update_fields(modified_tok.fields)
        #     # tok.update_metadata(modified_tok.metadata)

        while any(not res.ready() for res in results):
            pass

        for r in results:
            r = r.get()
            for token_result in r:
                tok = doc[token_result['index']]
                tok.dep_ = token_result['deprel']
                tok.head = doc[token_result['head']]

        return doc

    @staticmethod
    def _init_worker(worker_kwargs):
        global _WORKER_MODULE
        _WORKER_MODULE = ParzuWorker(**worker_kwargs)

    @staticmethod
    def _process_worker(vocab, tokens, start, end, i, out_queue):
        doc = Doc(vocab).from_bytes(tokens)
        span = doc[start:end]
        global _WORKER_MODULE

        # def send_update(x):
        #     out_queue.put(('update', i, x))

        result = _WORKER_MODULE.__call__(span)
        return result

class ParzuWorker:

    def __init__(self, parzu_home='resources/ParZu'):
        sys.path.insert(0, str(parzu_home))
        from parzu_class import process_arguments, Parser

        self.opts = process_arguments(commandline=False)
        self.parser = Parser(self.opts, timeout=1000)

    def process_parzu(self, span: Span):
        newinput = []
        for sent in span.sents:
            sent_strs = []
            for tok in sent:
                sent_strs.append(tok.text + '\t' + tok.tag_)

            newinput.append("\n".join(sent_strs))
        reformatted_input = "\n\n".join(newinput)

        output = self.parser.main(
            reformatted_input, inputformat="tagged", outputformat="conll"
        )
        return output

    def __call__(self, span: Span):
        result = []
        it = more_itertools.peekable(iter(span))

        for processed_sent in self.process_parzu(span):
            index_of_first_token = it.peek().i
            for line in processed_sent.split('\n'):
                if line.strip() == '':
                    continue
                tok = next(it)
                (
                    _,
                    word,
                    lemma,
                    _,
                    pos,
                    feats,
                    head,
                    deprel,
                    deps,
                    misc,
                ) = line.strip().split("\t")
                if deprel == 'root':
                    head = None
                else:
                    head = int(line.split('\t')[6])

                result.append({'index': tok.i,
                               'head': tok.i if head is None else index_of_first_token + int(head) - 1,
                               'deprel': deprel})

        return result
