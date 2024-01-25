import copy
import itertools
import logging
import unicodedata

import more_itertools
import regex as re
import pyuegc
from more_itertools import mark_ends
from spacy import Vocab
from spacy.tokens import Doc, Token, Span
from typing import Iterable, Tuple

from tqdm import tqdm

logger = logging.getLogger(__name__)

IRREGULAR_CHARACTERS = re.compile(
    r'[^\P{dt}\p{dt=canonical}]|[^\p{Latin}\pN-"‚‘„“.?!,;:\-–—*()\[\]{}/\'«‹›»’+&%# \t\n]',
    flags=re.UNICODE | re.MULTILINE)


class SoMaJoTokenizer:

    def __init__(self, vocab: Vocab, normalize=True, check_characters=True, paragraph_separator=None,
                 section_pattern=None, is_pretokenized=False, is_presentencized=False, pbar=False):
        if not normalize and not is_pretokenized:
            raise ValueError(""" cannot instantiate SoMaJoTokenizer with normalize=False and is_pretokenized=False,
            since full tokenization implies NFKC normalization!""")

        self.vocab = vocab
        self.normalize = normalize
        self.check_characters = check_characters
        self.paragraph_separator = paragraph_separator
        self.section_pattern = section_pattern
        self.is_pretokenized = is_pretokenized
        self.is_presentencized = is_presentencized
        self.pbar = pbar
        from somajo import SoMaJo

        self.tokenizer = SoMaJo("de_CMC", split_camel_case=True, split_sentences=not self.is_presentencized,
                                character_offsets=True)
        Token.set_extension('is_para_start', default=None, force=True)
        Token.set_extension('is_section_start', default=None, force=True)
        Token.set_extension('orig', default=None, force=True)
        Token.set_extension('orig_offset', default=None, force=True)

    def normalize_text(self, original, remove_whitespace=True):
        text = re.sub(r'([AOUaou])\N{Combining Latin Small Letter E}', r'\1\N{Combining Diaeresis}', original)
        text = unicodedata.normalize('NFKC', text)

        if remove_whitespace and re.search('\p{Whitespace}', text):
            # remove any extended grapheme cluster that contains a whitespace
            oldtext = text
            egcs = pyuegc.EGC(text)
            text = ''.join(egc for egc in egcs if re.search('\p{Whitespace}', egc) is None)
            logger.warning(f'Token "{original}" has been normalized to "{oldtext}" but contains whitespace! Replacing with "{text}".')
        
        assert re.search('\p{Whitespace}', text) is None
        return text

    def handle_pretokenized_paragraph(self, para) -> Iterable[Tuple[bool, str, bool, int]]:
        if self.is_presentencized:
            for sentence_str in para.split('\n'):
                for is_first, _, tok in mark_ends(sentence_str.split(' ')):
                    yield is_first, tok, None, None
        else:
            tokens = para.split(' ')
            for sentence in self.tokenizer._sentence_splitter.split(tokens):
                for is_first, _, tok in mark_ends(sentence):
                    yield is_first, tok, None, None

    def tokenize_paragraph(self, para: str) -> Iterable[Tuple[bool, str, bool, int]]:
        if self.is_presentencized:
            for sentence_match in re.finditer(r'[^\n]+', para, flags=re.UNICODE | re.MULTILINE):
                sent = sentence_match.group()
                sent_offset = sentence_match.start()
                for is_first, _, token in mark_ends(itertools.chain.from_iterable(self.tokenizer.tokenize_text([sent]))):
                    text = sent[token.character_offset[0]:token.character_offset[1]]
                    yield is_first, text, token.space_after, sent_offset + token.character_offset[0]
        else:
            for sentence in self.tokenizer.tokenize_text([para]):
                for is_first, _, token in mark_ends(sentence):
                    text = para[token.character_offset[0]:token.character_offset[1]]
                    yield is_first, text, token.space_after, token.character_offset[0]


    def __call__(self, text: str) -> Doc:
        words = []
        spaces = []
        sent_starts = []
        offsets = []

        orig_words = []
        section_starts = set()
        para_starts = set()

        if self.pbar:
            pbar = tqdm(total=len(text), leave=False, unit='B', unit_scale=True, ncols=80, postfix='somajo_tokenizer')
        else:
            pbar = None

        for is_section_start, para_offset, para in self.to_paragraphs(text):
            para_starts.add(len(words))
            if is_section_start:
                section_starts.add(len(words))

            if self.is_pretokenized:
                tokens = self.handle_pretokenized_paragraph(para)
            else:
                tokens = self.tokenize_paragraph(para)

            for is_sentence_start, word, space, offset in tokens:
                if self.normalize:
                    normalized = self.normalize_text(word)
                    if normalized == '':
                        continue
                    words.append(normalized)
                    orig_words.append(word)
                else:
                    words.append(word)

                sent_starts.append(is_sentence_start)
                if space is not None:
                    spaces.append(space)
                if offset is not None:
                    offsets.append(para_offset + offset)

            if pbar:
                pbar.update(len(para))


        if spaces:
            doc = Doc(self.vocab, words=words, spaces=spaces, sent_starts=copy.copy(sent_starts))
        else:
            doc = Doc(self.vocab, words=words, sent_starts=copy.copy(sent_starts))

        if self.normalize:
            for tok, orig_word in zip(doc, orig_words):
                tok._.orig = orig_word

        if offsets:
            for tok, offset in zip(doc, offsets):
                tok._.orig_offset = offset

        if self.check_characters:
            irr = [unicodedata.name(x) for x in set(IRREGULAR_CHARACTERS.findall(str(doc)))]
            if len(irr) > 0:
                logger.warning(f'Found irregular characters: {", ".join(irr)}')

        if self.paragraph_separator:
            for tok in doc:
                tok._.is_para_start = tok.i in para_starts

        if self.section_pattern:
            for tok in doc:
                tok._.is_section_start = tok.i in section_starts

        return doc

    def to_paragraphs(self, text: str) -> Iterable[Tuple[bool, int, str]]:
        if self.paragraph_separator is None:
            if self.section_pattern is None:
                yield None, 0, text
            else:
                yield True, 0, text
        else:
            is_section_start = True
            offset = 0
            for separator_match in re.finditer(self.paragraph_separator, text, flags=re.UNICODE | re.MULTILINE):
                para = text[offset:separator_match.start()]

                if len(para) == 0:
                    offset = separator_match.end()
                    continue

                if self.section_pattern and re.fullmatch(self.section_pattern, para, flags=re.UNICODE | re.MULTILINE):
                    is_section_start = True
                    offset = separator_match.end()
                    continue

                if self.section_pattern:
                    yield is_section_start, offset, para
                else:
                    yield None, offset, para

                offset = separator_match.end()
                is_section_start = False

            # handle final paragraph
            para = text[offset:]
            if len(para) == 0:
                return
            if self.section_pattern and re.fullmatch(self.section_pattern, para, flags=re.UNICODE | re.MULTILINE):
                return

            if self.section_pattern:
                yield is_section_start, offset, para
            else:
                yield None, offset, para
