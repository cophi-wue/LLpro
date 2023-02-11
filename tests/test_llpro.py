import collections
import copy
import json
import re
import sys
from abc import abstractmethod
from pathlib import Path
from typing import TypeVar, List, Tuple, Sequence, Iterable

import more_itertools
import pandas
import spacy
from spacy import Vocab
from spacy.tokens import Doc
from transformers import BertTokenizer

import llpro


class LLproReproduction:

    @abstractmethod
    def get_input(self, doc_key: str) -> Doc:
        raise NotImplementedError()

    @abstractmethod
    def get_expected_output(self, doc_key: str):
        raise NotImplementedError()

    @abstractmethod
    def run_pipeline(self, doc: Doc):
        raise NotImplementedError()

    @abstractmethod
    def get_test_docnames(self) -> Sequence[str]:
        raise NotImplementedError()

    def test_component(self):
        print('', file=sys.stderr)
        for doc_key in self.get_test_docnames():
            doc = self.get_input(doc_key)
            expected_output = self.get_expected_output(doc_key)
            actual_output = self.run_pipeline(doc)

            self.assert_equal(actual_output, expected_output)

    def assert_equal(self, actual_output, expected_output):
        actual_output = list(actual_output)
        expected_output = list(expected_output)
        assert actual_output == expected_output

    @staticmethod
    def read_conll(file, field_mapping=None) -> Doc:
        if field_mapping is None:
            field_mapping = {}

        words = []
        sent_starts = []

        collected_fields = collections.defaultdict(list)

        for sentence_lines in more_itertools.split_at(file, lambda line: line.strip() == ''):
            start = True
            for token_line in sentence_lines:
                fields = token_line.strip().split('\t')
                words.append(fields[1])
                sent_starts.append(start)
                start = False

                for index, v in field_mapping.items():
                    collected_fields[v] = fields[index]

        doc = Doc(Vocab(), words=words, sent_starts=copy.copy(sent_starts))
        for key, values in collected_fields.items():
            for tok, v in zip(doc, values):
                setattr(tok, key, v)

        return doc


class TestEventClassificationComponent(LLproReproduction):

    def get_input(self, doc_key: str) -> Doc:
        with open(Path('tests') / Path('expected_outputs') / Path('eventclassify_' + doc_key)) as f:
            output_doc = json.load(f)
            doc_text = output_doc[0]['text']
            nlp = spacy.load('de_dep_news_trf', disable=['tagger', 'ner'])
            doc = nlp(doc_text)

            # we now manually add the verbal phrases (without tags) returned by the reference implementation,
            # since we cannot reproduce these: reference implementation uses Spacy's v3.3 parser, we use Spacy v3.5.
            Doc.set_extension('events', default=list())
            expected_event_spans = []
            for event in output_doc[0]['annotations']:
                spans = []
                for char_span in event['spans']:
                    char_span_obj = doc.char_span(char_span[0], char_span[1], alignment_mode='expand')
                    spans.append(doc[char_span_obj.start:char_span_obj.end])
                expected_event_spans.append(spans)
            doc._.events = expected_event_spans
            return doc

    def get_expected_output(self, doc_key: str):
        with open(Path('tests') / Path('expected_outputs') / Path('eventclassify_' + doc_key)) as f:
            output_doc = json.load(f)
            doc_text = output_doc[0]['text']
            for i, event in enumerate(output_doc[0]['annotations']):
                for span in event['spans']:
                    yield event['predicted'], doc_text[span[0]:span[1]]

    def run_pipeline(self, doc: Doc):
        nlp = spacy.blank("de")
        nlp.add_pipe('events_uhhlt')
        nlp(doc)

        for i, event in enumerate(doc._.events):
            for span in event:
                yield event.attrs["event_type"], span.text

    def get_test_docnames(self) -> Sequence[str]:
        return ['Fontane_Theodor_Effi_Briest']


class TestSceneSegmenterComponent(LLproReproduction):

    def get_input(self, doc_key: str) -> Doc:
        tokenizer = BertTokenizer.from_pretrained('deepset/gbert-large')
        with open(Path('tests') / Path('inputs') / Path(doc_key + '.json')) as f:
            input_doc = json.load(f)
            words = []
            sent_starts = []

            for s in input_doc['sentences']:
                sentence_str = input_doc['text'][s['begin']:s['end']]

                # reproduce tokenization of the reference implementation
                tokenized_sentence = tokenizer.basic_tokenizer.tokenize(sentence_str)
                words.extend(tokenized_sentence)
                sent_starts.extend([True] + [False] * (len(tokenized_sentence) - 1))

        return Doc(Vocab(), words=words, sent_starts=copy.copy(sent_starts))

    def get_expected_output(self, doc_key: str) -> List[Tuple[str, int, str]]:
        tokenizer = BertTokenizer.from_pretrained('deepset/gbert-large')
        with open(Path('tests') / Path('expected_outputs') / Path('scenesegmenter_' + doc_key)) as f:
            input_doc = json.load(f)

            for i, s in enumerate(input_doc['scenes']):
                scene_str = input_doc['text'][s['begin']:s['end']]

                # reproduce tokenization of the reference implementation
                tokenized_sentence = tokenizer.basic_tokenizer.tokenize(scene_str)
                for tok in tokenized_sentence:
                    yield tok, i, s['type']

    def run_pipeline(self, doc: Doc) -> List[Tuple[str, int, str]]:
        nlp = spacy.blank("de")
        nlp.add_pipe('scenes_stss_se')
        nlp(doc)

        for tok in doc:
            yield tok.text, tok._.scene.id, tok._.scene.label_

    def get_test_docnames(self) -> Sequence[str]:
        return ['stss_trial_data']


class TestCorefComponent(LLproReproduction):

    def get_input(self, doc_key: str) -> Doc:
        with open(Path('tests') / Path('inputs') / Path(doc_key + '.json')) as f:
            words = []
            sent_starts = []

            input_doc = json.load(f)
            for sent in input_doc:
                words.extend(sent)
                sent_starts.extend([True] + [False] * (len(sent) - 1))

        return Doc(Vocab(), words=words, sent_starts=copy.copy(sent_starts))

    def parse_droc_file(self, file):
        words = []
        clusters = collections.defaultdict(list)
        coref_stacks = collections.defaultdict(list)

        word_idx = -1
        for line in file:
            if line.strip() == '' or line.startswith('#'):
                continue

            word_idx += 1
            row = line.split()  # Columns for each token
            assert len(row) >= 12
            word = row[3]
            coref = row[-1]
            if coref != '-' and coref != '_':
                for part in coref.split('|'):
                    if part[0] == '(':
                        if part[-1] == ')':
                            cluster_id = int(part[1:-1])
                            clusters[cluster_id].append((word_idx, word_idx))
                        else:
                            cluster_id = int(part[1:])
                            coref_stacks[cluster_id].append(word_idx)
                    else:
                        cluster_id = int(part[:-1])
                        start = coref_stacks[cluster_id].pop()
                        clusters[cluster_id].append((start, word_idx))

            words.append(word)
        return words, clusters

    def get_expected_output(self, doc_key: str) -> Iterable[Tuple[str, List[int]]]:
        with open(Path('tests') / Path('expected_outputs') / Path('neuralcoref_' + doc_key)) as f:
            words, clusters_map = self.parse_droc_file(f)
            clusters_of_word = collections.defaultdict(list)

            for j, cluster in clusters_map.items():
                for span in cluster:
                    for i in range(span[0], span[1] + 1):
                        clusters_of_word[i].append(j)

            for i, word in enumerate(words):
                yield word, clusters_of_word[i]

    def run_pipeline(self, doc: Doc) -> Iterable[Tuple[str, List[int]]]:
        nlp = spacy.blank("de")
        nlp.add_pipe('coref_uhhlt')
        nlp(doc)
        for tok in doc:
            yield tok.text, list(sorted(cluster.attrs['id'] for cluster in tok._.coref_clusters))

    def get_test_docnames(self) -> Sequence[str]:
        return ['droc_tokenized_text']


class TestRedewiedergabeComponent(LLproReproduction):
    fields = ['direct', 'indirect', 'reported', 'freeIndirect']

    def get_input(self, doc_key: str) -> Doc:
        df = pandas.read_csv(str(Path('tests') / Path('inputs') / Path(doc_key + '.xmi.tsv')), sep='\t')

        words = list(df['tok'])
        sent_starts = list(x == 'yes' for x in df['sentstart'])
        doc = Doc(Vocab(), words=words, sent_starts=copy.copy(sent_starts))
        return doc

    def get_expected_output(self, doc_key: str) -> Iterable[Tuple[str, bool, bool, bool, bool]]:
        df = pandas.read_csv(str(Path('tests') / Path('expected_outputs') / Path('redewiedergabe_' + doc_key)),
                             sep='\t')
        for _, row in df.iterrows():
            yield (row['tok'],) + tuple(row[f + '_pred'] != 'x' for f in self.fields)

    def run_pipeline(self, doc: Doc) -> Iterable[Tuple[str, bool, bool, bool, bool]]:
        nlp = spacy.blank("de")
        nlp.add_pipe('speech_redewiedergabe')
        nlp(doc)
        for tok in doc:
            yield (tok.text,) + tuple(rw_type in tok._.speech for rw_type in self.fields)

    def get_test_docnames(self) -> Sequence[str]:
        return ['rwk_digbib_1201-3']


class TestFLERTComponent(LLproReproduction):

    def get_input(self, doc_key: str) -> Doc:
        with open(Path('tests') / Path('inputs') / Path(doc_key + '.conll'), 'r') as f:
            return LLproReproduction.read_conll(f)

    def get_expected_output(self, doc_key: str) -> Iterable[dict]:
        with open(Path('tests') / Path('expected_outputs') / Path('flair_' + doc_key), 'r') as f:
            for line in f:
                ent = {}
                sent, start, end, type = re.match(r'^(\d+) Span\[(\d+):(\d+)]:.*→ ([A-Z]+)', line).groups()
                ent['sentence'] = int(sent)
                ent['token_start'] = int(start)
                ent['token_end'] = int(end)
                ent['type'] = type
                yield ent

    def run_pipeline(self, doc: Doc) -> List[dict]:
        nlp = spacy.blank("de")
        nlp.add_pipe('ner_flair')
        nlp(doc)

        sentence_starts = []
        for sent in doc.sents:
            sentence_starts.append(sent.start)

        sentence_starts = sorted(sentence_starts)

        entities = []
        for ent in doc.ents:
            ent_dict = {}
            ent_dict['sentence'] = sentence_starts.index(doc[ent.start].sent.start)
            ent_dict['token_start'] = ent.start - doc[ent.start].sent.start
            ent_dict['token_end'] = ent.end - doc[ent.start].sent.start
            ent_dict['type'] = ent.label_
            entities.append(ent_dict)

        return list(sorted(entities, key=lambda x: (x['sentence'], x['token_start'])))

    def get_test_docnames(self) -> Sequence[str]:
        return ['novelette']


class TestSoMeWeTaComponent(LLproReproduction):

    def get_input(self, doc_key: str) -> Doc:
        with open(Path('tests') / Path('inputs') / Path(doc_key + '.conll'), 'r') as f:
            words = []
            sent_starts = []

            for sentence_lines in more_itertools.split_at(f, lambda line: line.strip() == ''):
                start = True
                for token_line in sentence_lines:
                    fields = token_line.strip().split('\t')
                    words.append(fields[1])
                    sent_starts.append(start)
                    start = False

        doc = Doc(Vocab(), words=words, sent_starts=copy.copy(sent_starts))
        return doc

    def get_expected_output(self, doc_key: str) -> List[Tuple[str, str]]:
        with open(Path('tests') / Path('expected_outputs') / Path('someweta_' + doc_key), 'r') as f:
            output = [tuple(line.strip().split()[:2]) for line in f if line.strip() != '']
            return output

    def run_pipeline(self, doc: Doc) -> List[Tuple[str, str]]:
        nlp = spacy.blank("de")
        nlp.add_pipe('tagger_someweta')
        nlp(doc)
        return [(tok.text, tok.tag_) for tok in doc]

    def get_test_docnames(self) -> Sequence[str]:
        return ['novelette']


class TestRNNTaggerComponent(LLproReproduction):

    def get_input(self, doc_key: str) -> Doc:
        with open(Path('tests') / Path('expected_outputs') / Path('rnntagger_' + doc_key), 'r') as f:
            words = []
            sent_starts = []

            for sentence_lines in more_itertools.split_at(f, lambda line: line.strip() == ''):
                start = True
                for token_line in sentence_lines:
                    fields = token_line.strip().split('\t')
                    words.append(fields[0])
                    sent_starts.append(start)
                    start = False

        doc = Doc(Vocab(), words=words, sent_starts=copy.copy(sent_starts))
        return doc

    def get_expected_output(self, doc_key: str) -> Iterable[Tuple[str, str, str, str]]:
        with open(Path('tests') / Path('expected_outputs') / Path('rnntagger_' + doc_key), 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                fields = line.strip().split()
                from llpro.components.tagger_rnntagger import from_tigertag
                from spacy.tokens import MorphAnalysis
                morph = from_tigertag(fields[1])
                morph = MorphAnalysis(Vocab(), morph)
                # lemma = fields[2] if fields[2] != '<unk>' else fields[0]
                yield fields[0], fields[1], str(morph), fields[2]

    def run_pipeline(self, doc: Doc) -> Iterable[Tuple[str, str, str, str]]:
        nlp = spacy.blank("de")
        nlp.add_pipe('tagger_rnntagger')
        nlp.add_pipe('lemma_rnntagger')
        nlp(doc)
        return [(tok.text, tok._.rnntagger_tag, str(tok.morph), tok.lemma_) for tok in doc]

    def get_test_docnames(self) -> Sequence[str]:
        return ['novelette']


class TestParZuComponent(LLproReproduction):

    def get_input(self, doc_key: str) -> Doc:
        with open(Path('tests') / Path('inputs') / Path(doc_key + '.conll'), 'r') as f:
            words = []
            sent_starts = []
            pos = []

            for sentence_lines in more_itertools.split_at(f, lambda line: line.strip() == ''):
                start = True
                for token_line in sentence_lines:
                    fields = token_line.strip().split('\t')
                    words.append(fields[1])
                    pos.append(fields[4])
                    sent_starts.append(start)
                    start = False

        doc = Doc(Vocab(), words=words, sent_starts=copy.copy(sent_starts))
        for tok, tok_tag in zip(doc, pos):
            tok.tag_ = tok_tag

        return doc

    def get_expected_output(self, doc_key: str) -> List[Tuple[str, str, int]]:
        output = []
        with open(Path('tests') / Path('expected_outputs') / Path('parzu_' + doc_key), 'r') as f:
            for sentence_lines in more_itertools.split_at(f, lambda line: line.strip() == ''):
                sent_start_idx = len(output)
                for token_line in sentence_lines:
                    fields = token_line.strip().split('\t')
                    head_id = int(fields[6])
                    if head_id == 0:
                        output.append((fields[1], fields[7], 0))
                    else:
                        output.append((fields[1], fields[7], head_id - 1 + sent_start_idx))

        return output

    def get_test_docnames(self) -> Sequence[str]:
        return ['novelette']

    def run_pipeline(self, doc: Doc) -> List[Tuple[str, str, int]]:
        nlp = spacy.blank("de")
        nlp.add_pipe('parser_parzu_parallelized', config={'num_processes': 1, 'tokens_per_process': 100000})
        nlp(doc)
        return [(tok.text, tok.dep_, tok.head.i) for tok in doc]
