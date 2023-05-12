import copy
import logging
import unicodedata

import more_itertools
import regex as re
from spacy import Vocab
from spacy.tokens import Doc, Token, Span
from typing import Iterable, Tuple

logger = logging.getLogger(__name__)

IRREGULAR_CHARACTERS = re.compile(
    r'[^\P{dt}\p{dt=canonical}]|[^\p{Latin}\pN-"‚‘„“.?!,;:\-–—*()\[\]{}/\'«‹›»’+&%# \t\n]',
    flags=re.UNICODE | re.MULTILINE)


class SoMaJoTokenizer:

    def __init__(self, vocab: Vocab, normalize=True, check_characters=True, paragraph_separator=None,
                 section_pattern=None, is_pretokenized=False, is_presentencized=False):
        self.vocab = vocab
        self.normalize = normalize
        self.check_characters = check_characters
        self.paragraph_separator = paragraph_separator
        self.section_pattern = section_pattern
        self.is_pretokenized = is_pretokenized
        self.is_presentencized = is_presentencized
        from somajo import SoMaJo

        self.tokenizer = SoMaJo("de_CMC", split_camel_case=True, split_sentences=not self.is_presentencized)
        Token.set_extension('is_para_start', default=None)
        Token.set_extension('is_section_start', default=None)

    def __call__(self, text: str) -> Doc:
        from somajo.utils import Token
        if self.normalize:
            text = re.sub(r'([AOUaou])\N{Combining Latin Small Letter E}', r'\1\N{Combining Diaeresis}', text)
            text = unicodedata.normalize('NFKC', text)

        if self.check_characters:
            irr = [unicodedata.name(x) for x in set(IRREGULAR_CHARACTERS.findall(text))]
            if len(irr) > 0:
                logger.warning(f'Found irregular characters: {", ".join(irr)}')

        words = []
        spaces = []
        sent_starts = []

        section_starts = set()
        para_starts = set()

        for is_section_start, para in self.to_paragraphs(text):
            para_starts.add(len(words))
            if is_section_start:
                section_starts.add(len(words))

            if self.is_presentencized:
                sentence_strs = text.split('\n')
                if self.is_pretokenized:
                    sentences = [s.split(' ') for s in sentence_strs]
                else:
                    sentences = [self.tokenizer.tokenize_text([s]) for s in sentence_strs]
            else:
                if self.is_pretokenized:
                    tokens = text.split(' ')
                    sentences = self.tokenizer._sentence_splitter.split(tokens)
                else:
                    sentences = self.tokenizer.tokenize_text([para])

            for sent in sentences:
                if not sent:
                    continue
                sent_starts.extend([True] + [False] * (len(sent) - 1))

                if type(sent[0]) is str:
                    words.extend(sent)
                elif type(sent[0]) is Token:
                    words.extend([tok.text for tok in sent])
                    spaces.extend([tok.space_after for tok in sent])

        if spaces:
            doc = Doc(self.vocab, words=words, spaces=spaces, sent_starts=copy.copy(sent_starts))
        else:
            doc = Doc(self.vocab, words=words, sent_starts=copy.copy(sent_starts))

        # TODO add original form to tok._.orig?

        if self.paragraph_separator:
            for tok in doc:
                tok._.is_para_start = tok.i in para_starts

        if self.section_pattern:
            for tok in doc:
                tok._.is_section_start = tok.i in section_starts

        return doc

    def to_paragraphs(self, text: str) -> Iterable[Tuple[bool, str]]:
        if self.paragraph_separator is None:
            if self.section_pattern is None:
                yield None, text
            else:
                yield True, text
        elif self.section_pattern is None:
            for para in re.split(self.paragraph_separator, text, flags=re.UNICODE | re.MULTILINE):
                yield None, para
        else:
            is_section_start = True
            for para in re.split(self.paragraph_separator, text, flags=re.UNICODE | re.MULTILINE):
                if re.fullmatch(self.section_pattern, para, flags=re.UNICODE | re.MULTILINE):
                    is_section_start = True
                    continue
                yield is_section_start, para
                is_section_start = False
