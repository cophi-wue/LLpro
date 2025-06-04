import logging
import sys
from pathlib import Path

import torch
import more_itertools
from typing import Iterable, Callable, Sequence

from spacy import Language
from spacy.tokens import Span, Doc, Token, SpanGroup
from tqdm import tqdm

from ..common import Module
from .. import LLPRO_RESOURCES_ROOT

logger = logging.getLogger(__name__)


def add_extension(cls, ext, **kwargs):
    if not cls.has_extension(ext):
        cls.set_extension(ext, **kwargs)


@Language.factory("coref_uhhlt",
                  assigns=['token._.in_coref', 'token._.coref_clusters', 'doc._.has_coref', 'doc._.coref_clusters'],
                  default_config={
                      'coref_home': 'uhh-lt-neural-coref',
                      'model': 'model_droc_incremental_no_segment_distance_May02_17-32-58_1800.bin',
                      'config_name': 'droc_incremental_no_segment_distance',
                      'split_method': 'none',
                      'pbar_opts': None,
                      'use_cuda': True,
                      'device_on_run': True})
def coref_uhhlt(nlp, name, coref_home, model, config_name, split_method, pbar_opts, use_cuda, device_on_run):
    add_extension(Doc, "has_coref", default=False)
    add_extension(Doc, "coref_clusters", default=list())
    # add_extension(Doc, "coref_resolved")
    # add_extension(Doc, "coref_scores")
    add_extension(Span, "is_coref", default=False)
    add_extension(Span, "coref_cluster", default=list())
    # add_extension(Span, "coref_scores")
    add_extension(Token, "in_coref", default=False)
    add_extension(Token, "coref_clusters", default=list())

    return CorefTagger(name=name, coref_home=coref_home, model=model, config_name=config_name,
                       split_method=split_method, pbar_opts=pbar_opts, use_cuda=use_cuda,
                       device_on_run=device_on_run)


class CorefTagger(Module):

    def __init__(self, name, coref_home='uhh-lt-neural-coref',
                 model='model_droc_incremental_no_segment_distance_May02_17-32-58_1800.bin',
                 config_name='droc_incremental_no_segment_distance', split_method='none', use_cuda=True, device_on_run=True,
                 pbar_opts=None):
        super().__init__(name, pbar_opts)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")
        self.coref_home = Path(LLPRO_RESOURCES_ROOT) / coref_home
        self.model_path = Path(LLPRO_RESOURCES_ROOT) / model
        self.device_on_run = device_on_run
        
        self.split_method = split_method
        sys.path.insert(0, str(self.coref_home))
        from neural_coref.tensorize import Tensorizer
        from transformers import BertTokenizer, ElectraTokenizer
        from neural_coref.model import CorefModel, IncrementalCorefModel

        # cf. resources/uhh-lt-neural-coref/torch_serve/model_handler.py
        self.config = self.initialize_config(config_name)
        # assert self.config['incremental']
        # self.model = IncrementalCorefModel(self.config, self.device)
        if self.config['incremental']:
            self.model = IncrementalCorefModel(self.config, self.device)
        else:
            self.model = CorefModel(self.config, self.device)
        self.tensorizer = Tensorizer(self.config)
        self.model.load_state_dict(torch.load(str(self.model_path), map_location='cpu'))
        self.model.eval()
        if not self.device_on_run:
            self.model.to(self.device)
            logger.info(f"{self.name} using device {next(self.model.parameters()).device}")
        self.window_size = 384  # fixed hyperparameter, should be read out of config from MAR file

        self.tensorizer = Tensorizer(self.config)
        if self.config['model_type'] == 'electra':
            self.tokenizer = ElectraTokenizer.from_pretrained(self.config['bert_tokenizer_name'],
                                                              strip_accents=False)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.config['bert_tokenizer_name'])

        # apparently without effect as neither "keep" nor "discard" are recognized
        if self.config['incremental']:
            self.tensorizer.long_doc_strategy = "keep"
        else:
            self.tensorizer.long_doc_strategy = "discard"

    def initialize_config(self, config_name):
        import pyhocon
        config = pyhocon.ConfigFactory.parse_file(str(self.coref_home / "experiments.conf"))[config_name]
        return config

    def before_run(self):
        self.model.to(self.device)
        logger.info(f"{self.name} using device {str(next(self.model.parameters()).device)}")

    def after_run(self):
        if self.device_on_run:
            self.model.to('cpu')
            torch.cuda.empty_cache()

    # cf. resources/uhh-lt-neural-coref/model.py
    def get_predictions_incremental(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                                    is_training, update_fn=None):
        max_segments = 5

        entities = None
        from neural_coref.entities import IncrementalEntities
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
        # return starts, ends, mention_to_cluster_id, predicted_clusters
        return predicted_clusters

    def get_predictions_c2f(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                            is_training, update_fn=None):
        output = self.model.get_predictions_and_loss(input_ids.to(self.device),
                                                     input_mask.to(self.device), speaker_ids.to(self.device),
                                                     sentence_len.to(self.device), genre.to(self.device),
                                                     sentence_map.to(self.device), is_training.to(self.device))
        _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = output
        predicted_clusters, _, _ = self.model.get_predicted_clusters(
            span_starts.cpu().numpy(),
            span_ends.cpu().numpy(),
            antecedent_idx.cpu().numpy(),
            antecedent_scores.detach().cpu().numpy()
        )
        if update_fn is not None:
            update_fn(0, torch.sum(input_mask))
        return predicted_clusters

    def _tensorize(self, sentences: Iterable[Span]):
        nested_list = [[tok.text for tok in sent] for sent in sentences]

        from neural_coref.preprocess import get_document
        document = get_document('_', nested_list, 'german', self.window_size, self.tokenizer, 'nested_list')
        _, example = self.tensorizer.tensorize_example(document, is_training=False)[0]

        token_map = self.tensorizer.stored_info['subtoken_maps']['_']
        tensorized = [torch.tensor(e) for e in example[:7]]
        return tensorized, token_map

    def process_batch(self, sentences: Sequence[Span], pbar: tqdm) -> Doc:
        tensorized, subtoken_map = self._tensorize(sentences)

        def my_update_fn(start, end):
            pbar.update(subtoken_map[end - 1] - 1 - subtoken_map[start])

        with torch.no_grad():
            if self.config['incremental']:
                predicted_clusters = self.get_predictions_incremental(*tensorized, update_fn=my_update_fn)
            else:
                predicted_clusters = self.get_predictions_c2f(*tensorized, update_fn=my_update_fn)

            clusters = []
            for cluster in predicted_clusters:
                spans = []
                for mention_start, mention_end in cluster:
                    tok_start_idx = subtoken_map[mention_start]
                    tok_end_idx = subtoken_map[mention_end + 1]
                    spans.append((tok_start_idx, tok_end_idx))
                    #spans.append(doc[tok_start_idx:tok_end_idx])
                #clusters.append(SpanGroup(doc, spans=spans, attrs={'id': i}))
                clusters.append(spans)
            return clusters

    def split_fn(self, sent_a: Span, sent_b: Span):
        if self.split_method == 'none':
            return False
        if self.split_method == 'section':
            return sent_b[0]._.is_section_start
        if self.split_method == 'scene':
            return sent_a[-1]._.scene.id != sent_b[0]._.scene.id

    def process(self, doc: Doc, pbar: tqdm):
        sentences = doc.sents
        clusters = []

        i = 0
        for batch in more_itertools.split_when(sentences, lambda a,b: self.split_fn(a, b)):
            batch_offset = batch[0][0].i
            predicted_batch_clusters  = self.process_batch(batch, pbar)
            for predicted_batch_cluster in predicted_batch_clusters:
                cluster = []
                for tok_start_id, tok_end_id in predicted_batch_cluster:
                    cluster.append(doc[batch_offset + tok_start_id:batch_offset + tok_end_id])
                clusters.append(SpanGroup(doc, spans=cluster, attrs={'id': i}))
                i = i + 1

        for token in doc:
            token._.coref_clusters = []

        doc._.has_coref = True
        doc._.coref_clusters = clusters
        for cluster in clusters:
            for mention in cluster:
                mention._.is_coref = True
                mention._.coref_cluster = cluster
                # mention._.coref_scores = None  # not used
                for token in mention:
                    token._.in_coref = True
                    if cluster not in token._.coref_clusters:
                        token._.coref_clusters.append(cluster)

        return doc
