import logging
import unicodedata

import regex as re
from spacy import Vocab
from spacy.tokens import Doc, Token
from typing import Iterable

IRREGULAR_CHARACTERS = re.compile(
    r'[^\P{dt}\p{dt=canonical}]|[^\p{Latin}\pN-"‚‘„“.?!,;:\-–—*()\[\]{}/\'«‹›»’+&%# \t\n]',
    flags=re.UNICODE | re.MULTILINE)


class SoMaJoTokenizer:

    def __init__(self, vocab: Vocab, normalize=True, check_characters=True, paragraph_separator=None):
        self.vocab = vocab
        self.normalize = normalize
        self.check_characters = check_characters
        self.paragraph_separator = paragraph_separator
        from somajo import SoMaJo

        self.tokenizer = SoMaJo("de_CMC", split_camel_case=True)
        Token.set_extension('is_para_start', default=False)

    def __call__(self, text: str) -> Doc:
        if self.normalize:
            text = unicodedata.normalize('NFKC', text)

        if self.check_characters:
            irr = [unicodedata.name(x) for x in set(IRREGULAR_CHARACTERS.findall(text))]
            if len(irr) > 0:
                logging.warning(f'Found irregular characters: {", ".join(irr)}')

        words = []
        spaces = []
        sent_starts = []
        para_starts = []

        for para in self.to_paragraphs(text):
            sentences = list(self.tokenizer.tokenize_text(paragraphs=[str(para)]))
            para_starts.extend([True] + [False]*(sum(map(len, sentences))-1))
            for sent in sentences:
                words.extend([tok.text for tok in sent])
                spaces.extend([tok.space_after for tok in sent])
                sent_starts.extend([True] + [False]*(len(sent)-1))

        doc = Doc(self.vocab, words=words, spaces=spaces, sent_starts=sent_starts)
        for tok, p in zip(iter(doc), para_starts):
            tok._.is_para_start = p

        return doc

    def to_paragraphs(self, text: str) -> Iterable[str]:
        if self.paragraph_separator is None:
            yield text
            return

        yield from re.split(self.paragraph_separator, text, flags=re.UNICODE | re.MULTILINE)
