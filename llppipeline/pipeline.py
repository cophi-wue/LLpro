import logging
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
import warnings
from pathlib import Path

import regex

from .common import *

IRREGULAR_CHARACTERS = regex.compile(r'[^\P{dt}\p{dt=canonical}]|[^\p{Latin}\pN-"‚‘„“.?!,;:\-–—*(){}/\'«‹›»’+&%# \t\n]', flags=regex.UNICODE | regex.MULTILINE)

class NLTKPunktTokenizer(Tokenizer):

    def __init__(self, normalize=True, check_characters=True):
        self.normalize = normalize
        self.check_characters = check_characters

        from nltk.tokenize import word_tokenize, sent_tokenize

        def myprocessor(myinput):
            sentences = sent_tokenize(myinput, language="german")
            for i, sent in enumerate(sentences):
                for word in word_tokenize(sent, language="german"):
                    yield Token(word=word, sentence=i)

        self.processor = myprocessor

    def tokenize(self, file: TextIO, filename: str) -> Iterable[Token]:
        i = 0
        myinput = file.read()

        if self.normalize:
            myinput = unicodedata.normalize('NFKC', myinput)

        if self.check_characters:
            irr = [unicodedata.name(x) for x in set(IRREGULAR_CHARACTERS.findall(myinput))]
            if len(irr) > 0:
                logging.warning(f'Found irregular characters in {filename}: {", ".join(irr)}')

        for tok in self.processor(myinput):
            tok.doc = filename
            tok.id = i
            yield tok
            i += 1


class SoMeWeTaTagger(Module):

    def __init__(self, model='resources/german_newspaper_2018-12-21.model'):
        from someweta import ASPTagger

        self.tagger = ASPTagger()
        self.tagger.load(model)

        def myprocessor(sent):
            return self.tagger.tag_sentence([tok.word for tok in sent])

        self.processor = myprocessor

    def process(self, tokens: Sequence[Token], **kwargs):
        for sentence in Token.get_sentences(tokens):
            tagged = self.processor(sentence)
            assert len(tagged) == len(tagged)
            for token, (tok, tag) in zip(sentence, tagged):
                assert token.word == tok
                token.pos = tag


class RNNTagger(Module):

    def __init__(self, rnntagger_home='resources/RNNTagger', write_pos=False, write_morph=True):
        self.rnntagger_home = Path(rnntagger_home)
        self.write_pos = write_pos
        self.write_morph = write_morph
        # self.write_lemma = write_lemma
        sys.path.insert(0, str(self.rnntagger_home))
        sys.path.insert(0, str(self.rnntagger_home / "PyNMT"))
        import torch
        from PyRNN.Data import Data
        import PyRNN.RNNTagger
        import PyRNN.CRFTagger

        self.vector_mappings = Data(str(self.rnntagger_home / "lib/PyRNN/german.io"))
        self.model = torch.load(str(self.rnntagger_home / "lib/PyRNN/german.rnn"))
        torch.cuda.set_device(0)
        self.model = self.model.cuda()
        self.model.eval()
        logging.info(f"RNNTagger using device {next(self.model.parameters()).device}")

        def annotate_sentence(model, data, words):
            # map words to numbers and create Torch variables
            fwd_charIDs, bwd_charIDs = data.words2charIDvec(words)
            fwd_charIDs = model.long_tensor(fwd_charIDs)
            bwd_charIDs = model.long_tensor(bwd_charIDs)

            # optional word embeddings
            word_embs = None if data.word_emb_size <= 0 else model.float_tensor(data.words2vecs(words))

            # run the model
            if type(model) is PyRNN.RNNTagger.RNNTagger:
                tagscores = model(fwd_charIDs, bwd_charIDs, word_embs)
                _, tagIDs = tagscores.max(dim=-1)
            elif type(model) is PyRNN.CRFTagger.CRFTagger:
                tagIDs = model(fwd_charIDs, bwd_charIDs, word_embs)
            else:
                sys.exit("Error in function annotate_sentence")

            tags = data.IDs2tags(tagIDs)
            return tags

        def myprocessor(iterable_of_sentences):
            for sent in iterable_of_sentences:
                tokens = [x.word for x in sent]
                tags = annotate_sentence(self.model, self.vector_mappings, tokens)
                for tok, tag in zip(tokens, tags):
                    yield tok, tag
            # _, tmp_tagged_path = tempfile.mkstemp(text=True)
            # with open(tmp_tagged_path, "w", encoding="utf-8") as tmpfile:
            #     for sent in iterable_of_sentences:
            #         tokens = [x.word for x in sent]
            #         tags = annotate_sentence(self.model, self.vector_mappings, tokens)
            #         for tok, tag in zip(tokens, tags):
            #             print(tok, tag, sep='\t', file=tmpfile)
            #     print('\n', file=tmpfile)
            #
            # result = subprocess.run(
            #     ["bash", "rnn-tagger-lemmatizer.sh", tmp_tagged_path],
            #     cwd='resources',
            #     capture_output=True,
            #     text=True,
            # )
            # os.remove(tmp_tagged_path)
            # return result.stdout

        self.processor = myprocessor

    def process(self, tokens: Sequence[Token], **kwargs):
        it = iter(tokens)
        for tok, tag in self.processor(Token.get_sentences(tokens)):
            maintag = tag.split(".")[0]
            # kleine korrektur
            stts = "$." if maintag == "$" else maintag

            token = next(it)
            assert tok == token.word
            if self.write_morph:
                # TODO systematischer parsen?
                token.morph = re.search(r'^[^\.]+\.(.*)$', tag).group(1) if '.' in tag else None
            if self.write_pos:
                token.pos = stts
