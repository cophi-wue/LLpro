import torch
import torch.nn as nn
from transformers import AutoModel
from neural_coref import util
import logging
from collections.abc import Iterable
from collections import defaultdict
import numpy as np
import torch.nn.init as init
from neural_coref import higher_order as ho
from neural_coref.entities import IncrementalEntities, GoldLabelStrategy
from torch import Tensor
from collections import OrderedDict


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class CorefModel(nn.Module):
    def __init__(self, config, device, num_genres=None):
        super().__init__()
        self.config = config
        self.eval_only = False
        self.device = device

        self.num_genres = num_genres if num_genres else len(config['genres'])
        self.max_seg_len = config['max_segment_len']
        self.max_span_width = config['max_span_width']
        assert config['loss_type'] in ['marginalized', 'hinge']

        # Model
        self.dropout = nn.Dropout(p=config['dropout_rate'])
        self.bert = AutoModel.from_pretrained(config['bert_pretrained_name_or_path'])

        self.bert_emb_size = self.bert.config.hidden_size
        self.span_emb_size = self.bert_emb_size * 3
        if config['use_features']:
            self.span_emb_size += config['feature_emb_size']
        self.pair_emb_size = self.span_emb_size * 3
        if config['use_metadata']:
            self.pair_emb_size += 2 * config['feature_emb_size']
        if config['use_features']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_segment_distance']:
            self.pair_emb_size += config['feature_emb_size']

        assert config["span_width_embedding_size"] >= self.max_span_width
        self.emb_span_width = self.make_embedding(config["span_width_embedding_size"]) if config['use_features'] else None
        self.emb_span_width_prior = self.make_embedding(config["span_width_embedding_size"]) if config['use_width_prior'] else None
        self.emb_antecedent_distance_prior = self.make_embedding(config['num_antecedent_distance_buckets']) if config['use_distance_prior'] else None
        self.emb_genre = self.make_embedding(self.num_genres)
        self.emb_same_speaker = self.make_embedding(2) if config['use_metadata'] else None
        self.emb_segment_distance = self.make_embedding(config['max_training_sentences']) if config['use_segment_distance'] else None
        self.emb_top_antecedent_distance = self.make_embedding(config['num_antecedent_distance_buckets'])
        self.emb_cluster_size = self.make_embedding(10) if config['fine_grained'] and config['higher_order'] == 'cluster_merging' else None

        self.mention_token_attn = self.make_ffnn(self.bert_emb_size, 0, output_size=1) if config['model_heads'] else None
        self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=1)
        self.span_width_score_ffnn = self.make_ffnn(config['feature_emb_size'], [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if config['use_width_prior'] else None
        self.coarse_bilinear = self.make_ffnn(self.span_emb_size, 0, output_size=self.span_emb_size)
        self.antecedent_distance_score_ffnn = self.make_ffnn(config['feature_emb_size'], 0, output_size=1) if config['use_distance_prior'] else None
        self.coref_score_ffnn = self.make_ffnn(self.pair_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if config['fine_grained'] else None

        self.gate_ffnn = self.make_ffnn(2 * self.span_emb_size, 0, output_size=self.span_emb_size) if config['fine_grained'] and config['coref_depth'] > 1 else None
        self.span_attn_ffnn = self.make_ffnn(self.span_emb_size, 0, output_size=1) if config['fine_grained'] and config['higher_order'] == 'span_clustering' else None
        self.cluster_score_ffnn = self.make_ffnn(3 * self.span_emb_size + config['feature_emb_size'], [config['cluster_ffnn_size']] * config['ffnn_depth'], output_size=1) if config['fine_grained'] and config['higher_order'] == 'cluster_merging' else None

        self.update_steps = 0  # Internal use for debug
        self.debug = False

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        new_shape = (self.config['num_antecedent_distance_buckets'], self.config['feature_emb_size'])
        if 'emb_antecedent_distance_prior.weight' in state_dict and state_dict['emb_antecedent_distance_prior.weight'].shape != new_shape:
            logger.warn('Saved embedding for distance is of different size, some values are initialized randomly.')
            for weight_name in ['emb_antecedent_distance_prior.weight', 'emb_top_antecedent_distance.weight']:
                state_dict[weight_name] = self.initialize_differently_sized_embedding_layer(
                    state_dict[weight_name],
                    new_shape[0],
                )
        new_shape = (self.config['max_training_sentences'], self.config['feature_emb_size'])
        emb_segment_distance_weight = 'emb_segment_distance.weight'
        if emb_segment_distance_weight in state_dict and state_dict[emb_segment_distance_weight].shape != new_shape:
            logger.warn('Saved embedding for segment distance is of different size, some values are initialized randomly.')
            state_dict[emb_segment_distance_weight] = self.initialize_differently_sized_embedding_layer(
                state_dict[emb_segment_distance_weight],
                new_shape[0],
            )

        state_dict.pop('bert.embeddings.position_ids')
        return super().load_state_dict(state_dict, strict=strict)

    def initialize_differently_sized_embedding_layer(self, old_tensor, new_input_size, std=0.02):
        old_shape = old_tensor.shape
        new_weight = torch.empty(
            (new_input_size, self.config["feature_emb_size"]),
            dtype=old_tensor.dtype
        )
        init.normal_(new_weight, std=std)
        new_weight[
            :min(old_shape[0], new_weight.shape[0]), :min(self.config["feature_emb_size"], new_weight.shape[1])
        ] = old_tensor[:new_weight.shape[0], :new_weight.shape[1]]
        return new_weight

    def make_embedding(self, dict_size, std=0.02):
        emb = nn.Embedding(dict_size, self.config['feature_emb_size'])
        init.normal_(emb.weight, std=std)
        return emb

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i-1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def get_params(self, named=False):
        bert_based_param, task_param, incremental_param = [], [], []
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            elif 'entity_representation_gate' in name:
                to_add = (name, param) if named else param
                incremental_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param, incremental_param

    def get_candidate_spans(self, num_words, sentence_map, gold_info, device='cpu'):
        sentence_indices = sentence_map  # [num tokens]
        candidate_starts = torch.unsqueeze(torch.arange(0, num_words, device=device), 1).repeat(1, self.max_span_width)
        candidate_ends = candidate_starts + torch.arange(0, self.max_span_width, device=device)
        candidate_start_sent_idx = sentence_indices[candidate_starts]
        candidate_end_sent_idx = sentence_indices[torch.min(candidate_ends, torch.tensor(num_words - 1, device=device))]
        candidate_mask = (candidate_ends < num_words) & (candidate_start_sent_idx == candidate_end_sent_idx)
        candidate_starts, candidate_ends = candidate_starts[candidate_mask], candidate_ends[candidate_mask]  # [num valid candidates]
        num_candidates = candidate_starts.shape[0]

        candidate_labels = None
        if gold_info is not None:
            same_start = (torch.unsqueeze(gold_info['gold_starts'], 1) == torch.unsqueeze(candidate_starts, 0))
            same_end = (torch.unsqueeze(gold_info['gold_ends'], 1) == torch.unsqueeze(candidate_ends, 0))
            same_span = (same_start & same_end).to(torch.long)
            candidate_labels = torch.matmul(torch.unsqueeze(gold_info['gold_mention_cluster_map'], 0).to(torch.float), same_span.to(torch.float))
            candidate_labels = torch.squeeze(candidate_labels.to(torch.long), 0)  # [num candidates]; non-gold span has label 0
        return {
            'candidate_starts': candidate_starts,
            'candidate_ends': candidate_ends,
            'num_candidates': num_candidates,
            'candidate_labels': candidate_labels,
        }

    def get_top_spans(self, candidate_span_emb, candidates, num_words, do_loss, device='cpu'):
        conf = self.config

        candidate_starts = candidates['candidate_starts']
        candidate_ends = candidates['candidate_ends']
        candidate_labels = candidates['candidate_labels']
        candidate_width_idx = candidate_ends - candidate_starts

        # Get span score
        candidate_mention_scores = torch.squeeze(self.span_emb_score_ffnn(candidate_span_emb), 1)
        if conf['use_width_prior']:
            width_score = torch.squeeze(self.span_width_score_ffnn(self.emb_span_width_prior.weight), 1)
            candidate_width_score = width_score[candidate_width_idx]
            candidate_mention_scores += candidate_width_score

        # Extract top spans
        candidate_idx_sorted_by_score = torch.argsort(candidate_mention_scores, descending=True).tolist()
        candidate_starts_cpu, candidate_ends_cpu = candidate_starts.tolist(), candidate_ends.tolist()
        num_top_spans = int(min(conf['max_num_extracted_spans'], conf['top_span_ratio'] * num_words))
        selected_idx_cpu = self._extract_top_spans(candidate_idx_sorted_by_score, candidate_starts_cpu, candidate_ends_cpu, num_top_spans)
        assert len(selected_idx_cpu) == num_top_spans
        selected_idx = torch.tensor(selected_idx_cpu, device=device)
        top_span_starts, top_span_ends = candidate_starts[selected_idx], candidate_ends[selected_idx]
        top_span_emb = candidate_span_emb[selected_idx]
        top_span_cluster_ids = candidate_labels[selected_idx] if do_loss else None
        top_span_mention_scores = candidate_mention_scores[selected_idx]
        return {
            'starts': top_span_starts,
            'ends': top_span_ends,
            'emb': top_span_emb,
            'cluster_ids': top_span_cluster_ids,
            'mention_scores': top_span_mention_scores,
            'candidate_mention_scores': candidate_mention_scores,
            'num_top_spans': num_top_spans,
        }

    def build_candidate_embeddings(self, mention_doc, candidate_spans, device='cpu'):
        candidate_ends = candidate_spans['candidate_ends']
        candidate_starts = candidate_spans['candidate_starts']
        num_candidates = candidate_spans['num_candidates']
        num_words = mention_doc.shape[0]

        # Get span embedding (these are the bert embeddings)
        span_start_emb, span_end_emb = mention_doc[candidate_starts], mention_doc[candidate_ends]
        # list of all embeddings relevant to the candidate span, they are concated to form the span embedding
        candidate_emb_list = [span_start_emb, span_end_emb]
        if self.config['use_features']:
            candidate_width_idx = candidate_ends - candidate_starts
            candidate_width_emb = self.emb_span_width(candidate_width_idx)
            candidate_width_emb = self.dropout(candidate_width_emb)
            candidate_emb_list.append(candidate_width_emb)
        # Use attended head or avg token
        candidate_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(num_candidates, 1)  # [num_candidates, num_words]
        candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts, 1)) & (candidate_tokens <= torch.unsqueeze(candidate_ends, 1))
        if self.config['model_heads']:
            token_attn = torch.squeeze(self.mention_token_attn(mention_doc), 1)
        else:
            token_attn = torch.ones(num_words, dtype=torch.float, device=device)  # Use avg if no attention
        # Attention is zeroed out where the candidate tokens are False, i.e. where the token is not part of the candidate span
        candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn, 0)
        candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)
        # Attention for each candidate, all tokens in the span are weighted using attention (or if model_heads is False unweighted)
        head_attn_emb = torch.matmul(candidate_tokens_attn, mention_doc)
        candidate_emb_list.append(head_attn_emb)
        return candidate_emb_list

    def forward(self, *input):
        return self.get_predictions_and_loss(*input)

    def get_predictions_and_loss(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                                 is_training, gold_starts=None, gold_ends=None, gold_mention_cluster_map=None):
        """ Model and input are already on the device """
        device = self.device
        conf = self.config

        do_loss = False
        if gold_mention_cluster_map is not None:
            assert gold_starts is not None
            assert gold_ends is not None
            do_loss = True

        # Get token emb
        mention_doc = self.bert(input_ids, attention_mask=input_mask)[0]  # [num seg, num max tokens, emb size]
        input_mask = input_mask.to(torch.bool)
        mention_doc = mention_doc[input_mask]
        speaker_ids = speaker_ids[input_mask]
        num_words = mention_doc.shape[0]

        if not self.eval_only and not self.training and sentence_len.shape[0] > conf['doc_max_segments']:
            logger.warn('Not predicting document longer than doc_max_segments')
            return [
                None,
                None,
                None,
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([[0.1]]),
            ]

        # Get candidate span
        if do_loss:
            gold_info = {
                'gold_starts': gold_starts,
                'gold_ends': gold_ends,
                'gold_mention_cluster_map': gold_mention_cluster_map,
            }
        else:
            gold_info = None
        candidate_spans = self.get_candidate_spans(num_words, sentence_map, gold_info, device=device)
        # num_candidates = candidate_spans['num_candidates']

        candidate_span_emb = torch.cat(self.build_candidate_embeddings(mention_doc, candidate_spans, device=device), dim=1)  # [num candidates, new emb size]

        top_spans = self.get_top_spans(candidate_span_emb, candidate_spans, num_words, do_loss=do_loss, device=device)
        top_span_starts, top_span_ends = top_spans['starts'], top_spans['ends']
        top_span_emb = top_spans['emb']
        top_span_cluster_ids = top_spans['cluster_ids']
        top_span_mention_scores = top_spans['mention_scores']
        num_top_spans = top_spans['num_top_spans']

        # Coarse pruning on each mention's antecedents
        max_top_antecedents = min(num_top_spans, conf['max_top_antecedents'])
        top_span_range = torch.arange(0, num_top_spans, device=device)
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0)
        antecedent_mask = (antecedent_offsets >= 1)
        pairwise_mention_score_sum = torch.unsqueeze(top_span_mention_scores, 1) + torch.unsqueeze(top_span_mention_scores, 0)
        source_span_emb = self.dropout(self.coarse_bilinear(top_span_emb))
        target_span_emb = self.dropout(torch.transpose(top_span_emb, 0, 1))
        pairwise_coref_scores = torch.matmul(source_span_emb, target_span_emb)
        pairwise_fast_scores = pairwise_mention_score_sum + pairwise_coref_scores
        pairwise_fast_scores += torch.log(antecedent_mask.to(torch.float))
        if conf['use_distance_prior']:
            distance_score = torch.squeeze(self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), 1)
            bucketed_distance = util.bucket_distance(antecedent_offsets, num_buckets=self.config['num_antecedent_distance_buckets'])
            antecedent_distance_score = distance_score[bucketed_distance]
            pairwise_fast_scores += antecedent_distance_score
        top_pairwise_fast_scores, top_antecedent_idx = torch.topk(pairwise_fast_scores, k=max_top_antecedents)
        top_antecedent_mask = util.batch_select(antecedent_mask, top_antecedent_idx, device)  # [num top spans, max top antecedents]
        top_antecedent_offsets = util.batch_select(antecedent_offsets, top_antecedent_idx, device)

        # Slow mention ranking
        if conf['fine_grained']:
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb = None, None, None, None
            if conf['use_metadata']:
                top_span_speaker_ids = speaker_ids[top_span_starts]
                top_antecedent_speaker_id = top_span_speaker_ids[top_antecedent_idx]
                same_speaker = torch.unsqueeze(top_span_speaker_ids, 1) == top_antecedent_speaker_id
                same_speaker_emb = self.emb_same_speaker(same_speaker.to(torch.long))
                genre_emb = self.emb_genre(genre)
                genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat(num_top_spans, max_top_antecedents, 1)
            if conf['use_segment_distance']:
                num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
                token_seg_ids = torch.arange(0, num_segs, device=device).unsqueeze(1).repeat(1, seg_len)
                token_seg_ids = token_seg_ids[input_mask]
                top_span_seg_ids = token_seg_ids[top_span_starts]
                top_antecedent_seg_ids = token_seg_ids[top_span_starts[top_antecedent_idx]]
                top_antecedent_seg_distance = torch.unsqueeze(top_span_seg_ids, 1) - top_antecedent_seg_ids
                top_antecedent_seg_distance = torch.clamp(top_antecedent_seg_distance, 0, self.config['max_training_sentences'] - 1)
                seg_distance_emb = self.emb_segment_distance(top_antecedent_seg_distance)
            if conf['use_features']:  # Antecedent distance
                top_antecedent_distance = util.bucket_distance(top_antecedent_offsets, num_buckets=self.config['num_antecedent_distance_buckets'])
                top_antecedent_distance_emb = self.emb_top_antecedent_distance(top_antecedent_distance)

            for depth in range(conf['coref_depth']):
                top_antecedent_emb = top_span_emb[top_antecedent_idx]  # [num top spans, max top antecedents, emb size]
                feature_list = []
                if conf['use_metadata']:  # speaker, genre
                    feature_list.append(same_speaker_emb)
                    feature_list.append(genre_emb)
                if conf['use_segment_distance']:
                    feature_list.append(seg_distance_emb)
                if conf['use_features']:  # Antecedent distance
                    feature_list.append(top_antecedent_distance_emb)
                target_emb = torch.unsqueeze(top_span_emb, 1).repeat(1, max_top_antecedents, 1)
                similarity_emb = target_emb * top_antecedent_emb
                if len(feature_list) > 0:
                    feature_emb = torch.cat(feature_list, dim=2)
                    feature_emb = self.dropout(feature_emb)
                    pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)
                else:
                    pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb], 2)
                top_pairwise_slow_scores = torch.squeeze(self.coref_score_ffnn(pair_emb), 2)
                top_pairwise_scores = top_pairwise_slow_scores + top_pairwise_fast_scores
                if conf['higher_order'] == 'cluster_merging':
                    cluster_merging_scores = ho.cluster_merging(top_span_emb, top_antecedent_idx, top_pairwise_scores, self.emb_cluster_size, self.cluster_score_ffnn, None, self.dropout,
                                                                device=device, reduce=conf['cluster_reduce'], easy_cluster_first=conf['easy_cluster_first'])
                    break
                elif depth != conf['coref_depth'] - 1:
                    if conf['higher_order'] == 'attended_antecedent':
                        refined_span_emb = ho.attended_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores, device)
                    elif conf['higher_order'] == 'max_antecedent':
                        refined_span_emb = ho.max_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores, device)
                    elif conf['higher_order'] == 'entity_equalization':
                        refined_span_emb = ho.entity_equalization(top_span_emb, top_antecedent_emb, top_antecedent_idx, top_pairwise_scores, device)
                    elif conf['higher_order'] == 'span_clustering':
                        refined_span_emb = ho.span_clustering(top_span_emb, top_antecedent_idx, top_pairwise_scores, self.span_attn_ffnn, device)

                    gate = self.gate_ffnn(torch.cat([top_span_emb, refined_span_emb], dim=1))
                    gate = torch.sigmoid(gate)
                    top_span_emb = gate * refined_span_emb + (1 - gate) * top_span_emb  # [num top spans, span emb size]
        else:
            top_pairwise_scores = top_pairwise_fast_scores  # [num top spans, max top antecedents]

        if not do_loss:
            if conf['fine_grained'] and conf['higher_order'] == 'cluster_merging':
                top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)  # [num top spans, max top antecedents + 1]
            return [
                candidate_spans["candidate_starts"],
                candidate_spans["candidate_ends"],
                top_spans["mention_scores"],
                top_spans["starts"],
                top_spans["ends"],
                top_antecedent_idx,
                top_antecedent_scores,
            ]

        # Get gold labels
        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedent_idx]
        top_antecedent_cluster_ids += (top_antecedent_mask.to(torch.long) - 1) * 100000  # Mask id on invalid antecedents
        same_gold_cluster_indicator = (top_antecedent_cluster_ids == torch.unsqueeze(top_span_cluster_ids, 1))
        non_dummy_indicator = torch.unsqueeze(top_span_cluster_ids > 0, 1)
        pairwise_labels = same_gold_cluster_indicator & non_dummy_indicator
        dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))
        top_antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1)

        # Get loss
        top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)
        if conf['loss_type'] == 'marginalized':
            log_marginalized_antecedent_scores = torch.logsumexp(top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
            log_norm = torch.logsumexp(top_antecedent_scores, dim=1)
            loss = torch.sum(log_norm - log_marginalized_antecedent_scores)
        elif conf['loss_type'] == 'hinge':
            top_antecedent_mask = torch.cat([torch.ones(num_top_spans, 1, dtype=torch.bool, device=device), top_antecedent_mask], dim=1)
            top_antecedent_scores += torch.log(top_antecedent_mask.to(torch.float))
            highest_antecedent_scores, highest_antecedent_idx = torch.max(top_antecedent_scores, dim=1)
            gold_antecedent_scores = top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float))
            highest_gold_antecedent_scores, highest_gold_antecedent_idx = torch.max(gold_antecedent_scores, dim=1)
            slack_hinge = 1 + highest_antecedent_scores - highest_gold_antecedent_scores
            # Calculate delta
            highest_antecedent_is_gold = (highest_antecedent_idx == highest_gold_antecedent_idx)
            mistake_false_new = (highest_antecedent_idx == 0) & torch.logical_not(dummy_antecedent_labels.squeeze())
            delta = ((3 - conf['false_new_delta']) / 2) * torch.ones(num_top_spans, dtype=torch.float, device=device)
            delta -= (1 - conf['false_new_delta']) * mistake_false_new.to(torch.float)
            delta *= torch.logical_not(highest_antecedent_is_gold).to(torch.float)
            loss = torch.sum(slack_hinge * delta)

        # Add mention loss
        if conf['mention_loss_coef']:
            gold_mention_scores = top_span_mention_scores[top_span_cluster_ids > 0]
            loss_mention = -torch.sum(torch.log(torch.sigmoid(gold_mention_scores))) * conf['mention_loss_coef']
            loss += loss_mention

        if conf['higher_order'] == 'cluster_merging':
            top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)
            log_marginalized_antecedent_scores2 = torch.logsumexp(top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
            log_norm2 = torch.logsumexp(top_antecedent_scores, dim=1)  # [num top spans]
            loss_cm = torch.sum(log_norm2 - log_marginalized_antecedent_scores2)
            if conf['cluster_dloss']:
                loss += loss_cm
            else:
                loss = loss_cm

        # Debug
        if self.debug:
            if self.update_steps % 20 == 0:
                logger.info('---------debug step: %d---------' % self.update_steps)
                # logger.info('candidates: %d; antecedents: %d' % (num_candidates, max_top_antecedents))
                logger.info('spans/gold: %d/%d; ratio: %.2f' % (num_top_spans, (top_span_cluster_ids > 0).cpu().numpy().sum(), (top_span_cluster_ids > 0).cpu().numpy().sum()/num_top_spans))
                if conf['mention_loss_coef']:
                    logger.info('mention loss: %.4f' % loss_mention.item())
                if conf['loss_type'] == 'marginalized':
                    logger.info('norm/gold: %.4f/%.4f' % (torch.sum(log_norm).item(), torch.sum(log_marginalized_antecedent_scores).item()))
                else:
                    logger.info('loss: %.4f' % loss.item())
        self.update_steps += 1

        return [
            candidate_spans['candidate_starts'],
            candidate_spans['candidate_ends'],
            top_spans['candidate_mention_scores'],
            top_spans['starts'],
            top_spans['ends'],
            top_antecedent_idx,
            top_antecedent_scores
        ], loss

    def _extract_top_spans(self, candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans):
        """ Keep top non-cross-overlapping candidates ordered by scores; compute on CPU because of loop """
        selected_candidate_idx = []
        start_to_max_end, end_to_min_start = {}, {}
        for candidate_idx in candidate_idx_sorted:
            if len(selected_candidate_idx) >= num_top_spans:
                break
            # Perform overlapping check
            span_start_idx = candidate_starts[candidate_idx]
            span_end_idx = candidate_ends[candidate_idx]
            cross_overlap = False
            for token_idx in range(span_start_idx, span_end_idx + 1):
                max_end = start_to_max_end.get(token_idx, -1)
                if token_idx > span_start_idx and max_end > span_end_idx:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(token_idx, -1)
                if token_idx < span_end_idx and 0 <= min_start < span_start_idx:
                    cross_overlap = True
                    break
            if not cross_overlap:
                # Pass check; select idx and update dict stats
                selected_candidate_idx.append(candidate_idx)
                max_end = start_to_max_end.get(span_start_idx, -1)
                if span_end_idx > max_end:
                    start_to_max_end[span_start_idx] = span_end_idx
                min_start = end_to_min_start.get(span_end_idx, -1)
                if min_start == -1 or span_start_idx < min_start:
                    end_to_min_start[span_end_idx] = span_start_idx
        # Sort selected candidates by span idx
        selected_candidate_idx = sorted(selected_candidate_idx, key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))
        if len(selected_candidate_idx) < num_top_spans:  # Padding
            selected_candidate_idx += ([selected_candidate_idx[0]] * (num_top_spans - len(selected_candidate_idx)))
        return selected_candidate_idx

    def get_predicted_antecedents(self, antecedent_idx, antecedent_scores):
        """ CPU list input """
        predicted_antecedents = []
        for i, idx in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if idx < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents

    def get_predicted_clusters(self, span_starts, span_ends, antecedent_idx, antecedent_scores):
        """ CPU list input """
        # Get predicted antecedents
        predicted_antecedents = self.get_predicted_antecedents(antecedent_idx, antecedent_scores)

        # Get predicted clusters
        mention_to_cluster_id = {}
        predicted_clusters = []
        for i, predicted_idx in enumerate(predicted_antecedents):
            if predicted_idx < 0:
                continue
            assert i > predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx}'
            # Check antecedent's cluster
            antecedent = (int(span_starts[predicted_idx]), int(span_ends[predicted_idx]))
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1)
            if antecedent_cluster_id == -1:
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent])
                mention_to_cluster_id[antecedent] = antecedent_cluster_id
            # Add mention to cluster
            mention = (int(span_starts[i]), int(span_ends[i]))
            predicted_clusters[antecedent_cluster_id].append(mention)
            mention_to_cluster_id[mention] = antecedent_cluster_id

        predicted_clusters = [tuple(c) for c in predicted_clusters]
        return predicted_clusters, mention_to_cluster_id, predicted_antecedents

    def update_evaluator(self, span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator):
        predicted_clusters, mention_to_cluster_id, _ = self.get_predicted_clusters(span_starts, span_ends, antecedent_idx, antecedent_scores)
        mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters


class MentionModel(CorefModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = torch.nn.BCEWithLogitsLoss(torch.tensor(self.config["positive_class_weight"]))

    def forward(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                is_training, gold_starts=None, gold_ends=None, gold_mention_cluster_map=None):
        device = self.device
        conf = self.config

        if sentence_len.shape[0] > conf['doc_max_segments']:
            logger.warn('Not predicting document longer than doc_max_segments')
            return [
                torch.tensor([], device=device),
                torch.tensor([], device=device),
                {},
                torch.tensor(0.0, requires_grad=True),
            ]

        mention_doc = self.bert(input_ids, attention_mask=input_mask)[0]  # [num seg, num max tokens, emb size]

        input_mask = input_mask.to(torch.bool)
        mention_doc = mention_doc[input_mask]
        speaker_ids = speaker_ids[input_mask]

        num_words = mention_doc.shape[0]
        gold_info = {
            "gold_starts": gold_starts,
            "gold_ends": gold_ends,
            "gold_mention_cluster_map": gold_mention_cluster_map
        }
        candidates = self.get_candidate_spans(num_words, sentence_map, gold_info, device=device)

        span_start_emb, span_end_emb = mention_doc[candidates['candidate_starts']], mention_doc[candidates['candidate_ends']]
        candidate_emb_list = [span_start_emb, span_end_emb]

        if conf['use_features']:
            candidate_width_idx = candidates['candidate_ends'] - candidates['candidate_starts']
            candidate_width_emb = self.emb_span_width(candidate_width_idx)
            candidate_width_emb = self.dropout(candidate_width_emb)
            candidate_emb_list.append(candidate_width_emb)
        candidate_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(candidates['num_candidates'], 1)
        candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidates['candidate_starts'], 1)) & \
            (candidate_tokens <= torch.unsqueeze(candidates['candidate_ends'], 1))
        if conf['model_heads']:
            token_attn = torch.squeeze(self.mention_token_attn(mention_doc), 1)
        else:
            token_attn = torch.ones(num_words, dtype=torch.float, device=device)  # Use avg if no attention
        candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn, 0)
        candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)
        head_attn_emb = torch.matmul(candidate_tokens_attn, mention_doc)
        candidate_emb_list.append(head_attn_emb)
        candidate_span_emb = torch.cat(candidate_emb_list, dim=1)  # [num candidates, new emb size]
        scores = self.span_emb_score_ffnn(candidate_span_emb)
        loss = self.loss(scores.squeeze(), candidates["candidate_labels"].clone().to(torch.bool).to(torch.float))
        cluster_to_spans = defaultdict(list)
        new_cluster = 0
        for label, score, start, end in zip(
                candidates["candidate_labels"],
                scores,
                candidates["candidate_starts"],
                candidates["candidate_ends"],
        ):
            if score > 0.5:
                new_cluster += 1
                cluster_to_spans[new_cluster].append((start.item(), end.item()))

        return scores, candidates["candidate_labels"], cluster_to_spans, loss


class IncrementalCorefModel(CorefModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = torch.nn.CrossEntropyLoss()
        conf = args[0]
        # This is a really good idea in incremental models, uncomment at your peril
        # Otherwise only sub-sections of your documents are used, not great...
        assert conf['long_doc_strategy'] == 'keep'

        # Takes concat(entity_representation, span_representation)
        self.entity_representation_gate = nn.Linear(self.span_emb_size * 2, 1)
        # self.create_entity = torch.nn.Embedding(1, self.config['feature_emb_size'])

    def forward(self, *input, **kwargs):
        return self.get_predictions_incremental(*input, **kwargs)

    def update_evaluator(self, span_starts, span_ends, predicted_clusters, mention_to_cluster_id, gold_clusters, evaluator):
        mention_to_predicted = {m: tuple(predicted_clusters[cluster_idx]) for m, cluster_idx in mention_to_cluster_id.items()}
        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

    def get_predictions_incremental(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                                    is_training, gold_starts=None, gold_ends=None, gold_mention_cluster_map=None,
                                    global_loss_chance=0.0, teacher_forcing=False):
        max_segments = 5

        mention_to_cluster_id = {}
        predicted_clusters = []
        entities = None
        if torch.rand(1) < global_loss_chance:
            loss_strategy = GoldLabelStrategy.ORIGINAL
        else:
            loss_strategy = GoldLabelStrategy.MOST_RECENT
        cpu_entities = IncrementalEntities(conf=self.config, device="cpu", gold_strategy=loss_strategy)

        do_loss = False
        if gold_mention_cluster_map is not None:
            assert gold_starts is not None
            assert gold_ends is not None
            do_loss = True

        offset = 0
        total_loss = torch.tensor([0.0], requires_grad=True, device=self.device)
        total_gold_mask = None
        for i, start in enumerate(range(0, input_ids.shape[0], max_segments)):
            end = start + max_segments
            start_offset = torch.sum(input_mask[:start], (0, 1))
            delta_offset = torch.sum(input_mask[start:end], (0, 1))
            end_offset = start_offset + delta_offset
            if gold_ends is not None:
                # We include any gold spans that either end or start in the current window
                gold_mask = (gold_starts < end_offset) & (gold_starts > start_offset) | ((gold_ends > start_offset) & (gold_ends < end_offset))
                if total_gold_mask is None:
                    total_gold_mask = gold_mask
                else:
                    total_gold_mask |= gold_mask
                windowed_gold_starts = gold_starts[gold_mask].clamp(start_offset, end_offset) - start_offset
                windowed_gold_ends = gold_ends[gold_mask].clamp(start_offset, end_offset) - start_offset
                windowed_gold_mention_cluster_map = gold_mention_cluster_map[gold_mask]
            else:
                windowed_gold_starts = None
                windowed_gold_ends = None
                windowed_gold_mention_cluster_map = None
            res = self.get_predictions_incremental_internal(
                input_ids[start:end],
                input_mask[start:end],
                speaker_ids[start:end],
                sentence_len[start:end],
                genre,
                sentence_map[start_offset:end_offset],
                is_training,
                gold_starts=windowed_gold_starts,
                gold_ends=windowed_gold_ends,
                gold_mention_cluster_map=windowed_gold_mention_cluster_map,
                entities=entities,
                do_loss=do_loss,
                offset=offset,
                loss_strategy=loss_strategy,
                teacher_forcing=teacher_forcing,
            )
            offset += torch.sum(input_mask[start:end], (0, 1)).item()
            if do_loss:
                entities, new_cpu_entities, loss = res
                total_loss = loss + total_loss
            else:
                entities, new_cpu_entities = res
            cpu_entities.extend(new_cpu_entities)
        cpu_entities.extend(entities)
        starts, ends, mention_to_cluster_id, predicted_clusters = cpu_entities.get_result(
            remove_singletons=not self.config['incremental_singletons']
        )
        out = [
            starts,
            ends,
            mention_to_cluster_id,
            predicted_clusters,
        ]
        if do_loss:
            return out, loss
        else:
            return out

    def get_predictions_incremental_internal(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                                             is_training, gold_starts=None, gold_ends=None, gold_mention_cluster_map=None,
                                             entities=None, do_loss=None, offset=0, loss_strategy=GoldLabelStrategy.MOST_RECENT,
                                             teacher_forcing=False):
        device = self.device
        conf = self.config
        return_singletons = conf['incremental_singletons']

        # The model should already be trained so we detach the BERT part, massively improving performance
        mention_doc = self.bert(input_ids, attention_mask=input_mask)[0].detach()  # [num seg, num max tokens, emb size]

        input_mask = input_mask.to(torch.bool)
        mention_doc = mention_doc[input_mask]
        speaker_ids = speaker_ids[input_mask]

        num_words = mention_doc.shape[0]

        if do_loss:
            gold_info = {
                'gold_starts': gold_starts,
                'gold_ends': gold_ends,
                'gold_mention_cluster_map': gold_mention_cluster_map,
            }
            labels_for_starts = {(s.item(), e.item()): v.item() for s, e, v in zip(gold_starts, gold_ends, gold_mention_cluster_map)}
        else:
            gold_info = None
            labels_for_starts = {}

        candidate_spans = self.get_candidate_spans(num_words, sentence_map, gold_info, device=device)
        candidate_span_emb = torch.cat(self.build_candidate_embeddings(mention_doc, candidate_spans, device=device), dim=1)  # [num candidates, new emb size]

        top_spans = self.get_top_spans(candidate_span_emb, candidate_spans, num_words, do_loss=do_loss, device=device)
        top_span_starts, top_span_ends = top_spans['starts'], top_spans['ends']
        top_span_emb = top_spans['emb']

        new_cluster_threshold = torch.tensor([conf['new_cluster_threshold']]).unsqueeze(0).to(device)
        cpu_entities = IncrementalEntities(conf, "cpu", gold_strategy=loss_strategy)
        if entities is None:
            entities = IncrementalEntities(conf, device, gold_strategy=loss_strategy)

        if len(top_span_emb.shape) == 1:
            top_span_emb = top_span_emb.unsqueeze(0)

        losses = []
        cpu_loss = 0.0
        discard_weight = len(labels_for_starts.keys()) / len(top_span_starts)
        new_cluster_weight = len(set(labels_for_starts.keys())) / len(top_span_starts)
        for emb, span_start, span_end, mention_score in zip(top_span_emb, top_span_starts, top_span_ends, top_spans['mention_scores']):
            gold_class = labels_for_starts.get((span_start.item(), span_end.item()))
            if len(entities) == 0:
                # No need to do the whole similarity computation, this is the first mention
                entities.add_entity(emb, gold_class, span_start, span_end, offset=offset)
            else:
                if conf['evict']:
                    entities.evict(evict_to=cpu_entities)
                feature_list = []
                if conf['use_metadata']:
                    same_speaker_emb = torch.zeros(conf['feature_emb_size'], device=self.device)
                    genre_emb = self.emb_genre(genre)
                    feature_list.append(same_speaker_emb.repeat(entities.emb.shape[0]).reshape(-1, conf['feature_emb_size']))
                    feature_list.append(genre_emb.repeat(feature_list[0].shape[0]).reshape(-1, conf['feature_emb_size']))
                if conf['use_segment_distance']:
                    seg_distance_emb = self.emb_segment_distance(entities.sentence_distance.type(torch.long))
                    feature_list.append(seg_distance_emb.reshape(-1, conf['feature_emb_size']))
                if conf['use_features']:
                    dists = util.bucket_distance(entities.mention_distance, num_buckets=self.config['num_antecedent_distance_buckets'])
                    antecedent_distance_emb = self.emb_top_antecedent_distance(dists.type(torch.long))
                    feature_list.append(antecedent_distance_emb.reshape(-1, conf['feature_emb_size']))
                fast_source_span_emb = self.dropout(self.coarse_bilinear(emb))
                fast_entity_embs = self.dropout(torch.transpose(entities.emb, 0, 1))
                fast_coref_scores = torch.matmul(fast_source_span_emb, fast_entity_embs).unsqueeze(-1)
                feature_emb = torch.cat(feature_list, dim=1)
                feature_emb = self.dropout(feature_emb)
                embs = emb.repeat(entities.emb.shape[0], 1)
                similarity_emb = embs * entities.emb
                pair_emb = torch.cat([embs, entities.emb, similarity_emb, feature_emb], 1)
                # It's important for us to also involve the mention span scores, the only way to prune discovered spans is by turning them into singleton clusters
                # This is encouraged by explicitly involving the sum of the mention scores
                original_scores = self.coref_score_ffnn(pair_emb) + fast_coref_scores
                if return_singletons:
                    scores = torch.cat([new_cluster_threshold, -mention_score.view(1, 1), original_scores])
                else:
                    scores = torch.cat([new_cluster_threshold, original_scores])
                dist = torch.softmax(scores, 0)

                index_to_update = dist.argmax()
                if return_singletons:
                    cluster_to_update = index_to_update - 2
                else:
                    cluster_to_update = index_to_update - 1
                weights = torch.ones(scores.squeeze().T.shape)
                if return_singletons:
                    weights[0] = new_cluster_weight
                    weights[1] = discard_weight
                else:
                    weights[0] = discard_weight
                cre_loss = torch.nn.CrossEntropyLoss(weight=weights.to(self.device))
                if gold_class and do_loss:
                    if return_singletons:
                        target = torch.tensor([entities.class_gold_entity.get(gold_class, -2) + 2]).to(device)
                    else:
                        target = torch.tensor([entities.class_gold_entity.get(gold_class, -1) + 1]).to(device)
                    loss = cre_loss(scores.T, target)
                    losses.append(loss)
                elif do_loss:
                    # In this case we are training but don't have a gold label for this span
                    # i.e. the span is not in any gold cluster!
                    if return_singletons:
                        # In this case we add the option to discard a mention
                        loss = cre_loss(scores.T, torch.tensor([1], device=device))
                    else:
                        # Always create a new singleton cluster hoping nothing else ever gets added
                        loss = cre_loss(scores.T, torch.tensor([0], device=device))
                    losses.append(loss)
                if util.cuda_allocated_memory(self.device) > conf['memory_limit'] and len(losses) > 0:
                    sum(losses).backward(retain_graph=True)
                    cpu_loss += sum(losses).item()
                    losses = []
                if teacher_forcing:
                    forced_class = entities.class_gold_entity.get(gold_class)
                    if forced_class is None:
                        cluster_to_update = None
                    else:
                        cluster_to_update = torch.tensor(forced_class)
                    if cluster_to_update is None:
                        index_to_update = 0
                if index_to_update == 0:
                    entities.add_entity(emb, gold_class, span_start, span_end, offset=offset)
                elif index_to_update == 1 and return_singletons:
                    pass
                else:
                    update_gate = torch.sigmoid(
                        self.entity_representation_gate(torch.cat(
                            [
                                emb,
                                entities.emb[cluster_to_update],
                            ],
                        )))
                    entities.update_entity(
                        cluster_to_update,
                        emb,
                        gold_class,
                        span_start,
                        span_end,
                        update_gate,
                        offset=offset
                    )

        if len(losses) > 0:
            sum(losses).backward(retain_graph=True)
            cpu_loss += sum(losses).item()
        if do_loss:
            return entities, cpu_entities, cpu_loss
        else:
            return entities, cpu_entities
