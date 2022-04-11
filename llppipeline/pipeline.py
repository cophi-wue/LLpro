import pickle
import re
import sys
import unicodedata
from pathlib import Path

import regex

from .common import *

IRREGULAR_CHARACTERS = regex.compile(r'[^\P{dt}\p{dt=canonical}]|[^\p{Latin}\pN-"‚‘„“.?!,;:\-–—*()\[\]{}/\'«‹›»’+&%# \t\n]',
                                     flags=regex.UNICODE | regex.MULTILINE)


class NLTKPunktTokenizer(Tokenizer):

    def __init__(self, normalize=True, check_characters=True):
        self.normalize = normalize
        self.check_characters = check_characters

        from nltk.tokenize import word_tokenize, sent_tokenize

        def myprocessor(myinput):
            sentences = sent_tokenize(myinput, language="german")
            for i, sent in enumerate(sentences):
                for word in word_tokenize(sent, language="german"):
                    tok = Token()
                    tok.set_field('word', self.name, word)
                    tok.set_field('sentence', self.name, i)
                    yield tok

        self.processor = myprocessor

    def tokenize(self, content: str, filename: str = None) -> Iterable[Token]:
        i = 0
        if self.normalize:
            content = unicodedata.normalize('NFKC', content)

        if self.check_characters:
            irr = [unicodedata.name(x) for x in set(IRREGULAR_CHARACTERS.findall(content))]
            if len(irr) > 0:
                logging.warning(f'Found irregular characters in {filename}: {", ".join(irr)}')

        cursent = None
        for tok in self.processor(content):
            if cursent != tok.sentence:
                cursent = tok.sentence
                i = 1
            tok.set_field('id', self.name, i)
            if filename is not None:
                tok.set_field('doc', self.name, filename)
            yield tok
            i += 1


class SoMeWeTaTagger(Module):

    def __init__(self, model='resources/german_newspaper_2020-05-28.model'):
        from someweta import ASPTagger

        self.tagger = ASPTagger()
        self.tagger.load(model)

        def myprocessor(sent):
            return self.tagger.tag_sentence([tok.word for tok in sent])

        self.processor = myprocessor

    def process(self, tokens: Sequence[Token], update_fn, **kwargs):
        for sentence in Token.get_sentences(tokens):
            tagged = self.processor(sentence)
            assert len(tagged) == len(tagged)
            for token, (tok, tag) in zip(sentence, tagged):
                assert token.word == tok
                token.set_field('pos', self.name, tag)
                update_fn(1)


class RNNTagger(Module):

    def __init__(self, rnntagger_home='resources/RNNTagger', use_cuda=True):
        self.rnntagger_home = Path(rnntagger_home)
        sys.path.insert(0, str(self.rnntagger_home))
        sys.path.insert(0, str(self.rnntagger_home / "PyNMT"))
        import torch
        from PyRNN.Data import Data
        import PyRNN.RNNTagger
        import PyRNN.CRFTagger

        with open(str(self.rnntagger_home / "lib/PyRNN/german.hyper"), "rb") as file:
            hyper_params = pickle.load(file)
        self.vector_mappings = Data(str(self.rnntagger_home / "lib/PyRNN/german.io"))
        self.model = PyRNN.CRFTagger.CRFTagger(*hyper_params) if len(hyper_params) == 10 \
            else PyRNN.RNNTagger.RNNTagger(*hyper_params)
        self.model.load_state_dict(torch.load(str(self.rnntagger_home / "lib/PyRNN/german.rnn")))
        if torch.cuda.is_available() and use_cuda:
            self.model = self.model.cuda()
        self.model.eval()
        logging.info(f"RNNTagger using device {next(self.model.parameters()).device}")

        def annotate_sentence(model, data, words):
            # vgl. RNNTagger/PyRNN/rnn-annotate.py
            fwd_charIDs, bwd_charIDs = data.words2charIDvec(words)
            fwd_charIDs = model.long_tensor(fwd_charIDs)
            bwd_charIDs = model.long_tensor(bwd_charIDs)

            word_embs = None if data.word_emb_size <= 0 else model.float_tensor(data.words2vecs(words))

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

        self.processor = myprocessor

    def process(self, tokens: Sequence[Token], update_fn, **kwargs):
        it = iter(tokens)
        for tok, tag in self.processor(Token.get_sentences(tokens)):
            maintag = tag.split(".")[0]
            stts = "$." if maintag == "$" else maintag

            token = next(it)
            assert tok == token.word
            # TODO systematischer parsen?
            morph = re.search(r'^[^\.]+\.(.+)$', tag).group(1) if '.' in tag and not stts.startswith('$') else None
            token.set_field('morph', self.name, morph)
            token.set_field('pos', self.name, stts)
            update_fn(1)


class RNNLemmatizer(Module):
    def __init__(self, rnntagger_home='resources/RNNTagger', pos_module='RNNTagger', morph_module='RNNTagger',
                 use_cuda=True):
        self.rnntagger_home = Path(rnntagger_home)
        self.pos_module = pos_module
        self.morph_module = morph_module
        sys.path.insert(0, str(self.rnntagger_home))
        sys.path.insert(0, str(self.rnntagger_home / "PyNMT"))
        import torch
        from PyNMT.Data import Data, rstrip_zeros
        from PyNMT.NMT import NMTDecoder

        beam_size = 0
        batch_size = 32
        self.vector_mappings = Data(str(self.rnntagger_home / "lib/PyNMT/german.io"), batch_size)

        with open(str(self.rnntagger_home / "lib/PyNMT/german.hyper"), "rb") as file:
            hyper_params = pickle.load(file)
        self.model = NMTDecoder(*hyper_params)
        self.model.load_state_dict(torch.load(str(self.rnntagger_home / "lib/PyNMT/german.nmt")))
        if torch.cuda.is_available() and use_cuda:
            self.model = self.model.cuda()
        self.model.eval()
        logging.info(f"RNNLemmatizer using device {next(self.model.parameters()).device}")

        def process_batch(batch):
            # see RNNTagger/PyNMT/nmt-translate.py
            src_words, sent_idx, (src_wordIDs, src_len) = batch
            tgt_wordIDs, _ = self.model.translate(src_wordIDs, src_len, beam_size)
            # undo the sorting of sentences by length
            tgt_wordIDs = [tgt_wordIDs[i] for i in sent_idx]

            for swords, twordIDs in zip(src_words, tgt_wordIDs):
                twords = self.vector_mappings.target_words(rstrip_zeros(twordIDs))
                yield ''.join(twords)

        def format_batch(tokens):
            # see RNNTagger/scripts/reformat.pl
            batch = []
            for tok in tokens:
                word = tok.word
                if tok.get_field('morph', morph_module):
                    tag = tok.get_field('pos', pos_module) + '.' + tok.get_field('morph', morph_module)
                    # TODO geht das auch unabhängiger von dem direkten Morphologie-Format?
                else:
                    tag = tok.get_field('pos', pos_module)
                word = re.sub(r'   ', ' <> ', re.sub(r'(.)', r'\g<1> ', word))
                tag = re.sub(r'(.)', r'\g<1> ', tag)

                formatted = word + ' ## ' + tag + '\n'
                batch.append(formatted.split())
            return self.vector_mappings.build_test_batch(batch)

        def myprocessor(tokens_batch):
            assert len(tokens_batch) <= self.vector_mappings.batch_size
            for out in process_batch(format_batch(list(tokens_batch))):
                yield out

        self.processor = myprocessor

    def process(self, tokens: Sequence[Token], update_fn, **kwargs):
        for document in Token.get_documents(tokens):
            # for each document, use a cache to skip tokens already lemmatized
            cached = {}
            it = iter(document)
            done = False

            while not done:
                current_batch = []
                current_batch_is_cached = []
                try:
                    while sum(1 - x for x in current_batch_is_cached) < self.vector_mappings.batch_size:
                        tok = next(it)
                        cache_key = (
                            tok.word, tok.get_field('pos', self.pos_module), tok.get_field('morph', self.morph_module))
                        current_batch.append(tok)
                        if cache_key in cached.keys():
                            current_batch_is_cached.append(1)
                        else:
                            current_batch_is_cached.append(0)
                except StopIteration:
                    done = True
                    pass

                lemmas = iter(self.processor(
                    [tok for tok, is_cached in zip(current_batch, current_batch_is_cached) if not is_cached]))
                for tok, is_cached in zip(current_batch, current_batch_is_cached):
                    cache_key = (
                        tok.word, tok.get_field('pos', self.pos_module), tok.get_field('morph', self.morph_module))
                    if is_cached:
                        lemma = cached[cache_key]
                        tok.set_field('lemma', self.name, lemma)
                    else:
                        lemma = next(lemmas)
                        cached[cache_key] = lemma
                        tok.set_field('lemma', self.name, lemma)
                    update_fn(1)


class ParzuParser(Module):

    def __init__(self, parzu_home='resources/ParZu', pos_source=None):
        sys.path.insert(0, str(parzu_home))
        from parzu_class import process_arguments, Parser

        self.opts = process_arguments(commandline=False)
        self.parser = Parser(self.opts, timeout=1000)

        def myprocessor(document):
            newinput = []
            for sent in Token.get_sentences(document):
                sent_strs = []
                for tok in sent:
                    sent_strs.append(tok.word + '\t' + tok.get_field('pos', pos_source))

                newinput.append("\n".join(sent_strs))
            reformatted_input = "\n\n".join(newinput)

            output = self.parser.main(
                reformatted_input, inputformat="tagged", outputformat="conll"
            )
            return output

        self.processor = myprocessor

    def process(self, tokens: Sequence[Token], update_fn, **kwargs):
        for sent in Token.get_sentences(tokens):
            sent = list(sent)
            it = iter(sent)

            processed = self.processor(sent)
            assert len(processed) > 0
            for processed_sent in processed:
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
                        head = 0
                    else:
                        head = int(line.split('\t')[6])

                    assert tok.word == word
                    # if tok.pos != pos:
                    #     logging.info(
                    #         f'While processing ParZu: POS tags for token {tok.doc}, sentence {tok.sentence}, word {tok.id}'
                    #         f' "{tok.word}", POS tag {tok.pos} given vs {pos} generated by ParZu!')
                    #
                    tok.set_field('pos', self.name, pos)
                    tok.set_field('lemma', self.name, lemma)
                    tok.set_field('morph', self.name, feats)
                    tok.set_field('head', self.name, int(head))
                    tok.set_field('deprel', self.name, deprel)
                    update_fn(1)


class RedewiedergabeTagger(Module):

    def __init__(self, model_paths=None, use_cuda=True):
        import torch
        from flair.models import SequenceTagger

        self.model_paths = model_paths if model_paths is not None else \
            {'direct': 'resources/rwtagger_models/models/direct/final-model.pt',
             'indirect': 'resources/rwtagger_models/models/indirect/final-model.pt',
             'reported': 'resources/rwtagger_models/models/reported/final-model.pt',
             'freeIndirect': 'resources/rwtagger_models/models/freeIndirect/final-model.pt'}

        self.models: Dict[str, SequenceTagger] = {}
        for rw_type, model_path in self.model_paths.items():
            model = SequenceTagger.load(model_path)
            if torch.cuda.is_available() and use_cuda:
                model = model.cuda()
            model = model.eval()
            self.models[rw_type] = model
        logging.info(
            f"{self.name} using devices {','.join(str(next(m.parameters()).device) for m in self.models.values())}")

    def process(self, tokens: Sequence[Token], update_fn: Callable[[int], None], **kwargs) -> None:
        from flair.data import Sentence
        max_seq_length = 512

        def get_chunks():
            for sentence in Token.get_sentences(tokens):
                for chunk in Token.get_chunks(sentence, max_chunk_len=max_seq_length,
                                              sequence_length_function=lambda x: self.bert_sequence_length(x),
                                              borders='tokens'):
                    yield chunk

        for chunk in get_chunks():
            chunk = list(chunk)
            pred = [{} for _ in chunk]
            for rw_type, model in self.models.items():
                sent_obj = Sentence([tok.word for tok in chunk])
                model.predict(sent_obj)
                for i, labeled_token in enumerate(sent_obj.to_dict('cat')['entities']):
                    pred[i][rw_type] = labeled_token['labels'][0].to_dict()
                    pred[i][rw_type]['value'] = 'no' if pred[i][rw_type]['value'] == 'x' else 'yes'

            for tok, p in zip(chunk, pred):
                tok.set_field('redewiedergabe', self.name, p)
            update_fn(len(chunk))

    def bert_sequence_length(self, seq):
        from flair.embeddings import BertEmbeddings

        for model in self.models.values():
            if type(model.embeddings) == BertEmbeddings:
                return sum(len(model.embeddings.tokenizer.tokenize(tok.word)) for tok in seq)
        return len(seq)
