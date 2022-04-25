import logging
import os
import sys
from collections import defaultdict, OrderedDict
from typing import List

from objects import (
    WordOut,
    DocumentOut,
    PredicateOut,
    ArgumentOut,
    Doc,
    DocumentIn,
    AnnotationOut,
)
from sapienzanlp.predictors.srl import SemanticRoleLabeler

logger = logging.getLogger(__name__)


class Invero:
    def __init__(self, device, model_name, languages):
        # env params
        # device = os.getenv("DEVICE", "cpu")
        # model_name = os.getenv("MODEL_NAME", "resources/model")
        # lang_detector_path = os.getenv("LANG_DETECTOR", "resources/lid.176.bin")
        # languages = os.getenv("LANGUAGES", "en")
        self.languages = languages.lower().split()
        # for l in self.languages:
        #     if l.lower() not in SUPPORTED_LANGUAGES:
        #         logger.error(
        #             f"Language {l} not supported. For a list of supported languages please visit "
        #             f"http://verbatlas.org/api-documentation"
        #         )
        #         sys.exit(4)
        self.inventories = ["ca", "cs", "de", "en", "es", "va", "zh"]
        self.batch_size = int(os.getenv("BATCH_SIZE", 8))
        # models stuff
        # logger.info(f"Device in use: {device}")
        self.srl_model = SemanticRoleLabeler.from_pretrained(
            model_name, device=device, split_on_spaces=True
        )
        # additional stuff
        # self.language_detector = LanguageDetector(lang_detector_path)
        # self.sent_splitters = {
        #     lang: SpacySentenceSplitter(lang.lower(), "statistical") for lang in self.languages
        # }
        # self.tokenizers = {
        #     lang: SpacyTokenizer(lang.lower())
        #     if lang.lower() in SPACY_LANGUAGE_MAPPER
        #     else StanzaTokenizer(lang.lower())
        #     for lang in self.languages
        # }

    def __call__(self, docs: List[Doc], progress_fn=None) -> List[DocumentOut]:
        # preprocess text
        # docs = self.preprocess_text(sentences_in)
        # get batches and predict labels
        for batch in self.generator(docs):
            model_outputs = self.srl_model([b.tokens for b in batch], is_split_into_words=True)
            if progress_fn is not None:
                progress_fn(sum(len(b.tokens) for b in batch))
            for inventory, outputs in model_outputs.items():
                for b, output in zip(batch, outputs):
                    b.annotations[inventory] = output
        return self.clean_output(docs)

    # def preprocess_text(self, sentences_in: List[DocumentIn]) -> List[Doc]:
    #     """
    #     Preprocess the raw text in input.
    #     As of now, it uses spacy to detect the language and split the raw text in sentences.
    #     Then it uses stanza to pos-tag and lemmatize it. This is the fastest and most
    #     accurate way to do it.
    #
    #     Args:
    #         sentences_in (:obj:`List[str]`):
    #             A batch of text to preprocess.
    #
    #     Returns:
    #         :obj:`List[Doc]`: The text in input tokenized, pos-tagged and lemmatized.
    #
    #     """
    #     # flat the sentences in the document
    #     docs = self.flat_texts(sentences_in)
    #     # preprocess the sentences (tokenization, pos, lemma) in one big batch :)
    #     outputs = []
    #     # group batch by languages (may be different from one sentence
    #     # to the other
    #     preprocess_batch = []
    #     prev_lang = docs[0].lang
    #     for doc in docs:
    #         if doc.lang == prev_lang:
    #             preprocess_batch.append(doc)
    #         else:
    #             outputs += self.tokenizers[prev_lang]([d.text for d in preprocess_batch])
    #             preprocess_batch = [doc]
    #             prev_lang = doc.lang
    #     # left over
    #     outputs += self.tokenizers[prev_lang]([d.text for d in preprocess_batch])
    #     for doc, output in zip(docs, outputs):
    #         doc.tokens = output
    #     # split the docs longer than the maximum allowed from the LM
    #     docs = self.check_max_model_len(docs)
    #     return docs
    #
    # def flat_texts(self, sentences_in: List[DocumentIn]) -> List[Doc]:
    #     """
    #     Flat sentences in the input.
    #
    #     Args:
    #         sentences_in (:obj:`List[SentenceIn]`):
    #             Text batch in input.
    #
    #     Returns:
    #         :obj:`List[Doc]`: Flattened batch in input.
    #     """
    #     docs = []
    #     for doc_id, sentence_in in enumerate(sentences_in):
    #         if sentence_in.lang is None:
    #             sentence_in.lang = self.language_detector(sentence_in.text)
    #         sentence_in.lang = sentence_in.lang.lower()
    #         if sentence_in.lang not in self.languages:
    #             raise ValueError(f"Language ({sentence_in.lang}) not supported")
    #         # sentence segmentation
    #         sents = self.sent_splitters[sentence_in.lang](sentence_in.text, max_len=300)
    #         # add sentences to the output list
    #         for sid, sent in enumerate(sents):
    #             if sent:
    #                 docs.append(
    #                     Doc(
    #                         doc_id=doc_id,
    #                         sid=int(str(doc_id) + str(sid)),
    #                         lang=sentence_in.lang,
    #                         text=sent,
    #                     )
    #                 )
    #     return docs

    def clean_output(self, sentences: List[Doc]) -> List[DocumentOut]:
        """
        Clean the output to return (and groups back sentences from same document).

        Args:
            sentences (:obj:`List[Doc]`):
                Sentences in input

        Returns:
            :obj:`List[SentenceOut]`: the output to return to the caller

        """

        docs = defaultdict(DocumentOut)
        for sentence in sentences:
            word_index_doc_offset = len(docs[sentence.doc_id].tokens)
            for i, w in enumerate(sentence.tokens):
                docs[sentence.doc_id].tokens.append(
                    WordOut(index=word_index_doc_offset + i, raw_text=w.text)
                )

            annotation_dict = {}
            for inventory, annotations in sentence.annotations.items():
                for predicate in annotations.predicates:
                    predicate_index = predicate.index + word_index_doc_offset
                    arguments = [
                        ArgumentOut(
                            span=(
                                r.start_index + word_index_doc_offset,
                                r.end_index + word_index_doc_offset,
                            ),
                            role=r.role,
                            score=1.0,
                        )
                        for r in predicate.arguments
                        if r.role != "_"
                    ]
                    predicate_dict = annotation_dict.get(predicate_index, {})
                    predicate_dict[inventory] = PredicateOut(
                        frame_name=predicate.sense, roles=arguments
                    )
                    annotation_dict[predicate_index] = predicate_dict

            annotation_dict = OrderedDict(sorted(annotation_dict.items()))
            for index, annotation in annotation_dict.items():
                annotation_out = AnnotationOut(token_index=index)
                if "va" in annotation:
                    annotation_out.verbatlas = annotation["va"]
                if "en" in annotation:
                    annotation_out.english_propbank = annotation["en"]
                if "zh" in annotation:
                    annotation_out.chinese_propbank = annotation["zh"]
                if "de" in annotation:
                    annotation_out.german_propbank = annotation["de"]
                if "cz" in annotation:
                    annotation_out.pdt_vallex = annotation["cz"]
                if "es" in annotation:
                    annotation_out.spanish_ancora = annotation["es"]
                if "ca" in annotation:
                    annotation_out.catalan_ancora = annotation["ca"]
                docs[sentence.doc_id].annotations.append(annotation_out)
        return list(docs.values())

    def generator(self, inputs: List[Doc]) -> List[Doc]:
        """
        Batch generator for neural models.

        Args:
            inputs (:obj:`List[Doc]`):
                Input to batch.

        Returns:
            :obj:`List[Doc]`: Batched input.

        """
        batch = []
        for x in inputs:
            batch.append(x)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        # yield leftovers
        if batch:
            yield batch

    @staticmethod
    def check_max_model_len(docs, limit: int = 500):
        # split docs with `len > limit` for the model
        docs_ = []
        for doc in docs:
            if len(doc.tokens) > limit:
                n_chunks = len(doc.tokens) % limit
                tokens = [doc.tokens[i : i + n_chunks] for i in range(0, len(doc.tokens), n_chunks)]
                docs_.extend(
                    [
                        Doc(
                            doc_id=doc.doc_id,
                            sid=doc.sid,
                            lang=doc.lang,
                            text=doc.text,
                            tokens=t,
                        )
                        for t in tokens
                    ]
                )
            else:
                docs_.append(doc)
        return docs_
