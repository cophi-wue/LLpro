import copy
import itertools
import logging
import pickle
import re
import sys
import unicodedata
from pathlib import Path
from typing import Union

import flair
import more_itertools
import regex
import torch
from flair.data import Sentence

from .common import *
from .stts2upos import conv_table as stts2upos

IRREGULAR_CHARACTERS = regex.compile(
    r'[^\P{dt}\p{dt=canonical}]|[^\p{Latin}\pN-"‚‘„“.?!,;:\-–—*()\[\]{}/\'«‹›»’+&%# \t\n]',
    flags=regex.UNICODE | regex.MULTILINE)


class NLTKPunktTokenizer(Tokenizer):

    def __init__(self, normalize=True, check_characters=True):
        self.normalize = normalize
        self.check_characters = check_characters

    def tokenize(self, content: str, filename: str = None) -> Iterable[Token]:
        from nltk.tokenize import word_tokenize, sent_tokenize
        if self.normalize:
            content = unicodedata.normalize('NFKC', content)

        if self.check_characters:
            irr = [unicodedata.name(x) for x in set(IRREGULAR_CHARACTERS.findall(content))]
            if len(irr) > 0:
                logging.warning(f'Found irregular characters in {filename}: {", ".join(irr)}')

        sentences = sent_tokenize(content, language="german")
        for sent_id, sent in enumerate(sentences):
            for word_id, word in enumerate(word_tokenize(sent, language="german")):
                # property of the TreebankWordTokenizer
                word = word.replace("``", '"').replace("''", '"')
                tok = Token()
                tok.set_field('word', self.name, word)
                tok.set_field('sentence', self.name, sent_id + 1)
                tok.set_field('id', self.name, word_id)
                if filename is not None:
                    tok.set_field('doc', self.name, filename)
                yield tok


class SoMaJoTokenizer(Tokenizer):

    def __init__(self, normalize=True, check_characters=True):
        self.normalize = normalize
        self.check_characters = check_characters
        from somajo import SoMaJo

        self.tokenizer = SoMaJo("de_CMC", split_camel_case=True)

    def tokenize(self, content: str, filename: str = None) -> Iterable[Token]:
        if self.normalize:
            content = unicodedata.normalize('NFKC', content)

        if self.check_characters:
            irr = [unicodedata.name(x) for x in set(IRREGULAR_CHARACTERS.findall(content))]
            if len(irr) > 0:
                logging.warning(f'Found irregular characters in {filename}: {", ".join(irr)}')

        sentences = self.tokenizer.tokenize_text(paragraphs=[content])
        for sent_id, sent in enumerate(sentences):
            for word_id, word in enumerate(sent):
                # TODO spans rekonstruieren?
                tok = Token()
                tok.set_field('word', self.name, word.text, space_after=word.space_after, token_class=word.token_class)
                tok.set_field('sentence', self.name, sent_id + 1)
                tok.set_field('id', self.name, word_id)
                if filename is not None:
                    tok.set_field('doc', self.name, filename)
                yield tok


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
                upos, ufeats = stts2upos[tag]
                token.set_field('pos', self.name, tag)
                token.set_field('upos', self.name, upos)
                update_fn(1)


class RNNTagger(Module):

    def __init__(self, rnntagger_home='resources/RNNTagger', use_cuda=True):
        self.rnntagger_home = Path(rnntagger_home)
        sys.path.insert(0, str(self.rnntagger_home))
        sys.path.insert(0, str(self.rnntagger_home / "PyNMT"))
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
        logging.info(f"{self.name} using device {next(self.model.parameters()).device}")

        def annotate_sentence(model, data, words):
            # vgl. RNNTagger/PyRNN/rnn-annotate.py
            fwd_charIDs, bwd_charIDs = data.words2charIDvec(words)
            fwd_charIDs = model.long_tensor(fwd_charIDs)
            bwd_charIDs = model.long_tensor(bwd_charIDs)

            word_embs = None if data.word_emb_size <= 0 else model.float_tensor(data.words2vecs(words))

            with torch.no_grad():
                if type(model) is PyRNN.RNNTagger.RNNTagger:
                    tagscores = model(fwd_charIDs, bwd_charIDs, word_embs)
                    softmax_probs = torch.nn.functional.softmax(tagscores, dim=-1)    # ae: added softmax transform to get meaningful probabilities
                    best_prob, tagIDs = softmax_probs.max(dim=-1)
                    tags = data.IDs2tags(tagIDs)
                    return [{'tag': t, 'prob': p.item()} for t, p in zip(tags, best_prob.cpu())]
                elif type(model) is PyRNN.CRFTagger.CRFTagger:
                    tagIDs = model(fwd_charIDs, bwd_charIDs, word_embs)
                    return [{'tag': t} for t in data.IDs2tags(tagIDs)]
                else:
                    raise RuntimeError("Error in function annotate_sentence")

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
            if 'prob' in tag.keys():
                tag, prob = tag['tag'], tag['prob']
            else:
                tag = tag['tag']
                prob = None
            maintag = tag.split(".")[0]
            stts = "$." if maintag == "$" else maintag

            token = next(it)
            assert tok == token.word
            # TODO systematischer parsen?
            morph = re.search(r'^[^\.]+\.(.+)$', tag).group(1) if '.' in tag and not stts.startswith('$') else None
            upos, ufeats = stts2upos[stts]
            if prob is not None:
                token.set_field('morph', self.name, morph, prob=prob)
                token.set_field('pos', self.name, stts, prob=prob)
                token.set_field('upos', self.name, upos, prob=prob)
                # token.set_field('ufeats', self.name, ufeats, prob=prob)
            else:
                token.set_field('morph', self.name, morph)
                token.set_field('pos', self.name, stts)
                token.set_field('upos', self.name, upos)
                # token.set_field('ufeats', self.name, ufeats)
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
        logging.info(f"{self.name} using device {next(self.model.parameters()).device}")

        def process_batch(batch):
            # see RNNTagger/PyNMT/nmt-translate.py
            src_words, sent_idx, (src_wordIDs, src_len) = batch
            with torch.no_grad():
                tgt_wordIDs, tgt_logprobs = self.model.translate(src_wordIDs, src_len, beam_size)
            # undo the sorting of sentences by length
            tgt_wordIDs = [tgt_wordIDs[i] for i in sent_idx]

            for swords, twordIDs, logprob in zip(src_words, tgt_wordIDs, tgt_logprobs):
                twords = self.vector_mappings.target_words(rstrip_zeros(twordIDs))
                yield ''.join(twords), torch.exp(logprob).cpu().item()

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
            for out, prob in process_batch(format_batch(list(tokens_batch))):
                yield out, prob

        self.processor = myprocessor

    def process(self, tokens: Sequence[Token], update_fn, **kwargs):
        cached = {}
        it = iter(tokens)
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

            processed = iter(self.processor(
                [tok for tok, is_cached in zip(current_batch, current_batch_is_cached) if not is_cached]))
            for tok, is_cached in zip(current_batch, current_batch_is_cached):
                cache_key = (
                    tok.word, tok.get_field('pos', self.pos_module), tok.get_field('morph', self.morph_module))
                if is_cached:
                    lemma, prob = cached[cache_key]
                    tok.set_field('lemma', self.name, lemma, prob=prob)
                else:
                    lemma, prob = next(processed)
                    cached[cache_key] = (lemma, prob)
                    tok.set_field('lemma', self.name, lemma, prob=prob)
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

    def __init__(self, model_paths=None, use_cuda=True, device_on_run=False):
        import torch
        from flair.models import SequenceTagger
        flair.device = 'cpu'

        self.model_paths = model_paths if model_paths is not None else \
            {'direct': 'resources/rwtagger_models/models/direct/final-model.pt',
             'indirect': 'resources/rwtagger_models/models/indirect/final-model.pt',
             'reported': 'resources/rwtagger_models/models/reported/final-model.pt',
             'freeIndirect': 'resources/rwtagger_models/models/freeIndirect/final-model.pt'}
        self.device_on_run = device_on_run
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")

        self.models: Dict[str, SequenceTagger] = {}
        for rw_type, model_path in self.model_paths.items():
            model = SequenceTagger.load(model_path)
            model = model.eval()
            self.models[rw_type] = model

        if not self.device_on_run:
            for model in self.models.values():
                model.to(self.device)
            logging.info(
                f"{self.name} using devices {','.join(str(next(m.parameters()).device) for m in self.models.values())}")

    def before_run(self):
        flair.device = self.device
        if self.device_on_run:
            for model in self.models.values():
                model.to(self.device)
            logging.info(
                f"{self.name} using devices {','.join(str(next(m.parameters()).device) for m in self.models.values())}")

    def after_run(self):
        for model in self.models.values():
            model.to('cpu')
            torch.cuda.empty_cache()  # TODO

    def process(self, tokens: Sequence[Token], update_fn: Callable[[int], None], **kwargs) -> None:
        from flair.data import Sentence
        max_seq_length = 510  # inc. [CLS} and [SEP]

        def gen_sentences():
            for sentence in Token.get_sentences(tokens):
                for sent_part in Token.get_chunks(sentence, max_chunk_len=max_seq_length,
                                              sequence_length_function=lambda x: self.bert_sequence_length(x),
                                              borders='tokens'):
                    yield sent_part

        for chunk in more_itertools.chunked(gen_sentences(), n=10):
            chunk = [list(sent) for sent in chunk]
            for rw_type, model in self.models.items():
                sent_objs = [Sentence([tok.word for tok in sent]) for sent in chunk]
                with torch.no_grad():
                    model.predict(sent_objs)

                labels = itertools.chain.from_iterable(sent.to_dict('cat')['cat'] for sent in sent_objs)
                for label, tok in zip(labels, itertools.chain.from_iterable(chunk)):
                    value = 'no' if label['value'] == 'x' else 'yes'
                    tok.set_field(f'speech_{rw_type}', self.name, value, prob=label['confidence'])

            update_fn(sum(len(sent) for sent in chunk))

    def bert_sequence_length(self, seq):
        from flair.embeddings import BertEmbeddings

        for model in self.models.values():
            if type(model.embeddings) == BertEmbeddings:
                return sum(len(model.embeddings.tokenizer.tokenize(tok.word)) for tok in seq)
        return len(seq)


class FLERTNERTagger(Module):

    def __init__(self, mini_batch_size=8, use_cuda=True, device_on_run=False):
        self.device_on_run = device_on_run
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")

        from flair.models import SequenceTagger
        from flair.data import Sentence
        from flair.datasets import DataLoader
        from flair.datasets import FlairDatapointDataset
        # see https://github.com/flairNLP/flair/issues/2650#issuecomment-1063785119
        flair.device = 'cpu'
        self.tagger = SequenceTagger.load("flair/ner-german-large")

        def myprocessor(sentences: Sequence[Sentence]) -> Iterable[Sentence]:
            dataloader = DataLoader(
                dataset=FlairDatapointDataset(sentences),
                batch_size=mini_batch_size,
            )
            with torch.no_grad():
                for batch in dataloader:
                    self._annotate_batch(batch)
                    for sentence in batch:
                        yield sentence
                        sentence.clear_embeddings()

        self.processor = myprocessor

        if not self.device_on_run:
            self.tagger.to(self.device)
            logging.info(f"{self.name} using device {str(next(self.tagger.parameters()).device)}")

    def before_run(self):
        if self.device_on_run:
            flair.device = self.device
            self.tagger.to(self.device)
            logging.info(f"{self.name} using device {str(next(self.tagger.parameters()).device)}")

    def after_run(self):
        if self.device_on_run:
            self.tagger.to('cpu')
            torch.cuda.empty_cache()  # TODO

    def _annotate_batch(self, batch):
        from flair.models.sequence_tagger_utils.bioes import get_spans_from_bio
        import torch

        # cf. flair/models/sequence_tagger_model.py
        # get features from forward propagation
        features, gold_labels = self.tagger.forward(batch)

        # Sort batch in same way as forward propagation
        lengths = torch.LongTensor([len(sentence) for sentence in batch])
        _, sort_indices = lengths.sort(dim=0, descending=True)
        batch = [batch[i] for i in sort_indices]

        # make predictions
        if self.tagger.use_crf:
            predictions, all_tags = self.tagger.viterbi_decoder.decode(features, False)
        else:
            predictions, all_tags = self.tagger._standard_inference(features, batch, False)

        for sentence, sentence_predictions in zip(batch, predictions):
            if self.tagger.predict_spans:
                sentence_tags = [label[0] for label in sentence_predictions]
                sentence_scores = [label[1] for label in sentence_predictions]
                predicted_spans = get_spans_from_bio(sentence_tags, sentence_scores)
                for predicted_span in predicted_spans:
                    span = sentence[predicted_span[0][0]:predicted_span[0][-1] + 1]
                    span.add_label(self.tagger.tag_type, value=predicted_span[2], score=predicted_span[1])

            # token-labels can be added directly ("O" and legacy "_" predictions are skipped)
            else:
                for token, label in zip(sentence.tokens, sentence_predictions):
                    if label[0] in ["O", "_"]:
                        continue
                    token.add_label(typename=self.tagger.tag_type, value=label[0], score=label[1])

    def process(self, tokens: Sequence[Token], update_fn: Callable[[int], None], **kwargs) -> None:
        span_id = 0
        sentences = [list(s) for s in Token.get_sentences(tokens)]
        flair_sentences = [Sentence([tok.word for tok in s]) for s in sentences]
        for sentence, tagged_sentence in zip(sentences, self.processor(flair_sentences)):
            if self.tagger.predict_spans:
                result = [list() for _ in sentence]
                for span in tagged_sentence.get_spans('ner'):
                    span_id = span_id + 1
                    for tok in span.tokens:
                        result[tok.idx - 1].append((span.tag, span.score, span_id))
                for res, tok in zip(result, sentence):
                    if len(res) == 0:
                        continue
                    tags, scores, ids = list(zip(*res))
                    tok.set_field('ner', self.name, list(tags), scores=list(scores), ids=list(ids))
            else:
                for tok, tagged_tok in zip(sentence, tagged_sentence):
                    value = tagged_tok.get_label('ner').value
                    tok.set_field('ner', self.name, [value])
            update_fn(len(sentence))


class CorefIncrementalTagger(Module):

    def __init__(self, coref_home='resources/uhh-lt-neural-coref',
                 model='resources/model_droc_incremental_no_segment_distance_May02_17-32-58_1800.bin',
                 config_name='droc_incremental_no_segment_distance',
                 use_cuda=True,
                 device_on_run=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")
        self.coref_home = Path(coref_home)
        self.model_path = Path(model)
        self.device_on_run = device_on_run
        sys.path.insert(0, str(self.coref_home))
        from tensorize import Tensorizer
        from transformers import BertTokenizer, ElectraTokenizer
        from model import IncrementalCorefModel

        # cf. resources/uhh-lt-neural-coref/torch_serve/model_handler.py
        self.config = self.initialize_config(config_name)
        assert self.config['incremental']
        self.model = IncrementalCorefModel(self.config, self.device)
        self.tensorizer = Tensorizer(self.config)
        self.model.load_state_dict(torch.load(str(model), map_location='cpu'))
        self.model.eval()
        if not self.device_on_run:
            self.model.to(self.device)
            logging.info(f"{self.name} using device {next(self.model.parameters()).device}")
        self.window_size = 384  # fixed hyperparameter, should be read out of config from MAR file

        self.tensorizer = Tensorizer(self.config)
        if self.config['model_type'] == 'electra':
            self.tokenizer = ElectraTokenizer.from_pretrained(self.config['bert_tokenizer_name'],
                                                              strip_accents=False)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.config['bert_tokenizer_name'])

        # apparently without effect as neither "keep" nor "discard" are recognized
        self.tensorizer.long_doc_strategy = "keep"

    def initialize_config(self, config_name):
        import pyhocon
        config = pyhocon.ConfigFactory.parse_file(str(self.coref_home / "experiments.conf"))[config_name]
        return config

    def before_run(self):
        if self.device_on_run:
            self.model.to(self.device)
            logging.info(f"{self.name} using device {next(self.model.parameters()).device}")

    def after_run(self):
        if self.device_on_run:
            self.model.to('cpu')
            torch.cuda.empty_cache()  # TODO

    # cf. resources/uhh-lt-neural-coref/model.py
    def get_predictions_incremental(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                                    is_training, update_fn=None):
        max_segments = 5

        entities = None
        from entities import IncrementalEntities
        cpu_entities = IncrementalEntities(conf=self.config, device="cpu")

        offset = 0
        for i, start in enumerate(range(0, input_ids.shape[0], max_segments)):
            end = start + max_segments
            start_offset = torch.sum(input_mask[:start], (0, 1))
            delta_offset = torch.sum(input_mask[start:end], (0, 1))
            end_offset = start_offset + delta_offset
            res = self.model.get_predictions_incremental_internal(
                input_ids[start:end].to(self.device),
                input_mask[start:end].to(self.device),
                speaker_ids[start:end].to(self.device),
                sentence_len[start:end].to(self.device),
                genre.to(self.device),
                sentence_map[start_offset:end_offset].to(self.device),
                is_training.to(self.device),
                entities=entities,
                offset=offset,
            )

            if update_fn is not None:
                update_fn(start_offset, end_offset)

            offset += torch.sum(input_mask[start:end], (0, 1)).item()
            entities, new_cpu_entities = res
            cpu_entities.extend(new_cpu_entities)
        cpu_entities.extend(entities)
        starts, ends, mention_to_cluster_id, predicted_clusters = cpu_entities.get_result(
            remove_singletons=not self.config['incremental_singletons']
        )
        return starts, ends, mention_to_cluster_id, predicted_clusters

    def _tensorize(self, tokens):
        nested_list = [[tok.word for tok in sent] for sent in Token.get_sentences(tokens)]

        from preprocess import get_document
        document = get_document('_', nested_list, 'german', self.window_size, self.tokenizer, 'nested_list')
        _, example = self.tensorizer.tensorize_example(document, is_training=False)[0]

        token_map = self.tensorizer.stored_info['subtoken_maps']['_']
        tensorized = [torch.tensor(e) for e in example[:7]]
        return tensorized, token_map

    def process(self, tokens: Sequence[Token], update_fn: Callable[[int], None], **kwargs) -> None:
        tokens = list(tokens)
        mentions = [set() for _ in tokens]
        tensorized, subtoken_map = self._tensorize(tokens)

        def my_update_fn(start, end):
            update_fn(subtoken_map[end - 1] - 1 - subtoken_map[start])

        with torch.no_grad():
            _, _, _, predicted_clusters = self.get_predictions_incremental(*tensorized, update_fn=my_update_fn)
            for i, cluster in enumerate(predicted_clusters):
                for mention_start, mention_end in cluster:
                    for r in range(mention_start, mention_end + 1):
                        mentions[subtoken_map[r]].add(i)

            for mention_set, tok in zip(mentions, tokens):
                tok.set_field('coref_clusters', self.name, list(mention_set))


class InVeRoXL(Module):

    def __init__(self, inveroxl_home='resources/inveroxl', use_cuda=True, device_on_run=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")
        self.inveroxl_home = Path(inveroxl_home)
        self.device_on_run = device_on_run
        sys.path.insert(0, str(self.inveroxl_home))
        from inference import Invero

        self.model = Invero(device='cpu', model_name=str(self.inveroxl_home / "resources" / "model"),
                            languages='de')
        if not self.device_on_run:
            self.model.srl_model.device = self.device
            self.model.srl_model.model.to(self.device)
            logging.info(f"{self.name} using device {next(self.model.srl_model.model.parameters()).device}")

    def before_run(self):
        if self.device_on_run:
            self.model.srl_model.device = self.device
            self.model.srl_model.model.to(self.device)
            logging.info(f"{self.name} using device {next(self.model.srl_model.model.parameters()).device}")

    def after_run(self):
        if self.device_on_run:
            self.model.srl_model.model.to('cpu')
            self.model.srl_model.device = 'cpu'
            torch.cuda.empty_cache()  # TODO

    def prepare_docs(self, tokens):
        from sapienzanlp.data.model_io.word import Word
        from objects import Doc
        for i, sent in enumerate(Token.get_sentences(tokens)):
            prepared_tokens = [Word(text=w.word, index=j) for j, w in enumerate(sent)]
            yield Doc(doc_id=0, sid=i, lang='de', text=None, tokens=prepared_tokens)

    def process(self, tokens: Sequence[Token], update_fn: Callable[[int], None], **kwargs) -> None:
        tokens = list(tokens)
        prepared_docs = list(self.prepare_docs(tokens))
        prepared_docs = self.model.check_max_model_len(prepared_docs)
        out = self.model(prepared_docs, progress_fn=update_fn)
        assert len(out) == 1
        annotated_doc = out[0]
        assert len(annotated_doc.tokens) == len(tokens)

        frames = [list() for _ in tokens]
        annotations = [(an.token_index, an.verbatlas, an) for an in annotated_doc.annotations if
                       an.verbatlas.frame_name != '_']
        for i, (token_index, an, orig) in enumerate(annotations):
            if token_index >= len(tokens):
                logging.warning(f'InVeRo returns annotation out of bound: {orig}')
                continue
            frames[token_index].append({'id': i, 'sense': an.frame_name})
            for r in an.roles:
                for j in range(*r.span):
                    frames[j].append({'id': i, 'role': r.role})

        for tok_frames, tok in zip(frames, tokens):
            tok.set_field('srl', self.name, tok_frames)

#
# class TagsetTranslator(Module):
