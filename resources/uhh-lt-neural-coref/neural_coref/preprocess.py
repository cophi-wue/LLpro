import argparse
import logging
import os
import re
import collections
import json
from transformers import BertTokenizer, ElectraTokenizer
from neural_coref import conll, util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def skip_doc(doc_key):
    return False


def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def get_sentence_map(segments, sentence_end):
    assert len(sentence_end) == sum([len(seg) - 2 for seg in segments])  # of subtokens in all segments
    sent_map = []
    sent_idx, subtok_idx = 0, 0
    for segment in segments:
        sent_map.append(sent_idx)  # [CLS]
        for i in range(len(segment) - 2):
            sent_map.append(sent_idx)
            sent_idx += int(sentence_end[subtok_idx])
            subtok_idx += 1
        sent_map.append(sent_idx)  # [SEP]
    return sent_map


class DocumentState(object):
    def __init__(self, key):
        self.doc_key = key
        self.tokens = []

        # Linear list mapped to all subtokens
        self.subtokens = []
        self.subtoken_map = []
        self.token_end = []
        self.sentence_end = []
        self.info = []  # Only non-none for the first subtoken of each word

        # Linear list mapped to subtokens with CLS, SEP
        self.sentence_map = []

        # Segments (mapped to subtokens with CLS, SEP)
        self.segments = []
        self.segment_subtoken_map = []
        self.segment_info = []  # Only non-none for the first subtoken of each word
        self.speakers = []

        # Doc-level attributes
        self.pronouns = []
        self.clusters = collections.defaultdict(list)  # {cluster_id: [(first_subtok_idx, last_subtok_idx) for each mention]}
        self.coref_stacks = collections.defaultdict(list)

    def finalize(self, input_format):
        """ Extract clusters; fill other info e.g. speakers, pronouns """
        # Populate speakers from info
        subtoken_idx = 0
        for seg_info in self.segment_info:
            speakers = []
            for i, subtoken_info in enumerate(seg_info):
                if i == 0 or i == len(seg_info) - 1:
                    speakers.append('[SPL]')
                elif subtoken_info is not None:  # First subtoken of each word
                    speakers.append(subtoken_info[9])
                    if subtoken_info[4] == 'PRP':
                        self.pronouns.append(subtoken_idx)
                else:
                    speakers.append(speakers[-1])
                subtoken_idx += 1
            self.speakers += [speakers]

        # Populate cluster
        first_subtoken_idx = -1  # Index of the first subtoken of each word
        for seg_idx, seg_info in enumerate(self.segment_info):
            for i, subtoken_info in enumerate(seg_info):
                first_subtoken_idx += 1
                coref = subtoken_info[-2] if subtoken_info is not None else '-'
                if coref != '-' and coref != '_':
                    last_subtoken_idx = first_subtoken_idx + subtoken_info[-1] - 1
                    for part in coref.split('|'):
                        if part[0] == '(':
                            if part[-1] == ')':
                                cluster_id = int(part[1:-1])
                                self.clusters[cluster_id].append((first_subtoken_idx, last_subtoken_idx))
                            else:
                                cluster_id = int(part[1:])
                                self.coref_stacks[cluster_id].append(first_subtoken_idx)
                        else:
                            cluster_id = int(part[:-1])
                            try:
                                start = self.coref_stacks[cluster_id].pop()
                                self.clusters[cluster_id].append((start, last_subtoken_idx))
                            except IndexError:
                                # Since our splitting approach does not take into account entities this can, very occasionally happen.
                                logger.warning("Trying to close non-existent entity (this should only happen very rarely).")

        # Merge clusters if any clusters have common mentions
        merged_clusters = []
        for cluster in self.clusters.values():
            existing = None
            for mention in cluster:
                for merged_cluster in merged_clusters:
                    if mention in merged_cluster:
                        existing = merged_cluster
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often)")
                existing.update(cluster)
            else:
                merged_clusters.append(set(cluster))

        merged_clusters = [list(cluster) for cluster in merged_clusters]
        all_mentions = util.flatten(merged_clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = util.flatten(self.segment_subtoken_map)

        # Sanity check
        assert len(all_mentions) == len(set(all_mentions))  # Each mention unique
        # Below should have length: # all subtokens with CLS, SEP in all segments
        num_all_seg_tokens = len(util.flatten(self.segments))
        assert num_all_seg_tokens == len(util.flatten(self.speakers))
        assert num_all_seg_tokens == len(subtoken_map)
        assert num_all_seg_tokens == len(sentence_map)

        return {
            "doc_key": self.doc_key,
            "tokens": self.tokens,
            "sentences": self.segments,
            "speakers": self.speakers,
            "constituents": [],
            "ner": [],
            "clusters": merged_clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
            'pronouns': self.pronouns
        }


def split_into_segments(document_state: DocumentState, max_seg_len, constraints1, constraints2, tokenizer):
    """ Split into segments.
        Add subtokens, subtoken_map, info for each segment; add CLS, SEP in the segment subtokens
        Input document_state: tokens, subtokens, token_end, sentence_end, utterance_end, subtoken_map, info
    """
    curr_idx = 0  # Index for subtokens
    prev_token_idx = 0
    while curr_idx < len(document_state.subtokens):
        # Try to split at a sentence end point
        end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)  # Inclusive
        while end_idx >= curr_idx and not constraints1[end_idx]:
            end_idx -= 1
        if end_idx < curr_idx:
            logger.info(f'{document_state.doc_key}: no sentence end found; split at token end')
            # If no sentence end point, try to split at token end point
            end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)
            while end_idx >= curr_idx and not constraints2[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                logger.warning('Cannot split valid segment: no sentence end or token end, will split at subword')
                end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)

        segment = [tokenizer.cls_token] + document_state.subtokens[curr_idx: end_idx + 1] + [tokenizer.sep_token]
        document_state.segments.append(segment)

        subtoken_map = document_state.subtoken_map[curr_idx: end_idx + 1]
        document_state.segment_subtoken_map.append([prev_token_idx] + subtoken_map + [subtoken_map[-1]])

        document_state.segment_info.append([None] + document_state.info[curr_idx: end_idx + 1] + [None])

        curr_idx = end_idx + 1
        prev_token_idx = subtoken_map[-1]


def get_document(doc_key, doc_lines, language, seg_len, tokenizer, input_format):
    """ Process raw input to finalized documents """
    document_state = DocumentState(doc_key)
    word_idx = -1

    if input_format == 'conll-2012':
        word_col = 3
    elif input_format == 'semeval-2010':
        word_col = 1
    elif input_format == 'nested_list':
        word_col = 0
    else:
        raise Exception(f"Invalid input format '{input_format}'")

    # Build up documents
    if input_format != 'nested_list':
        for line in doc_lines:
            row = line.split()  # Columns for each token
            if len(row) == 0:
                document_state.sentence_end[-1] = True
            else:
                assert len(row) >= 12
                word_idx += 1
                word = normalize_word(row[word_col], language)
                subtokens = tokenizer.tokenize(word)
                document_state.tokens.append(word)
                document_state.token_end += [False] * (len(subtokens) - 1) + [True]
                for idx, subtoken in enumerate(subtokens):
                    document_state.subtokens.append(subtoken)
                    if idx != 0:
                        info = None
                    else:
                        info = (row + [len(subtokens)])
                    document_state.info.append(info)
                    document_state.sentence_end.append(False)
                    document_state.subtoken_map.append(word_idx)
    else:
        for sentence in doc_lines:
            for token in sentence:
                word_idx += 1
                word = normalize_word(token, language)
                subtokens = tokenizer.tokenize(word)
                document_state.tokens.append(word)
                document_state.token_end += [False] * (len(subtokens) - 1) + [True]
                for idx, subtoken in enumerate(subtokens):
                    document_state.subtokens.append(subtoken)
                    info = None
                    document_state.info.append(info)
                    document_state.sentence_end.append(False)
                    document_state.subtoken_map.append(word_idx)
            document_state.sentence_end[-1] = True

    # Split documents
    constraits1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
    split_into_segments(document_state, seg_len, constraits1, document_state.token_end, tokenizer)
    document = document_state.finalize(input_format)
    return document


def minimize_partition(partition, extension, args, tokenizer):
    input_path = os.path.join(args.input_dir, f'{partition}.{args.language}.{extension}')
    output_path = os.path.join(args.output_dir, f'{partition}.{args.language}.{args.seg_len}.jsonlines')
    doc_count = 0
    logger.info(f'Minimizing {input_path}...')

    # Read documents
    documents = []  # [(doc_key, lines)]
    with open(input_path, 'r') as input_file:
        for line in input_file.readlines():
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
                documents.append((doc_key, []))
            elif line.startswith('#end document'):
                continue
            else:
                documents[-1][1].append(line)

    # Write documents
    with open(output_path, 'w') as output_file:
        for doc_key, doc_lines in documents:
            if skip_doc(doc_key):
                continue
            document = get_document(doc_key, doc_lines, args.language, args.seg_len, tokenizer, args.input_format)
            output_file.write(json.dumps(document))
            output_file.write('\n')
            doc_count += 1
    logger.info(f'Processed {doc_count} documents to {output_path}')


def minimize_language(args):
    if args.model_type == "electra":
        tokenizer = ElectraTokenizer.from_pretrained(args.tokenizer_name, strip_accents=False)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)

    minimize_partition('dev', args.input_suffix, args, tokenizer)
    minimize_partition('test', args.input_suffix, args, tokenizer)
    minimize_partition('train', args.input_suffix, args, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-cased',
                        help='Name or path of the tokenizer/vocabulary')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory that contains conll files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--seg_len', type=int, default=128,
                        help='Segment length: 128, 256, 384, 512')
    parser.add_argument('--language', type=str, default='english',
                        help='english, chinese, arabic, german')
    parser.add_argument('--model_type', type=str, default='bert')
    # parser.add_argument('--lower_case', action='store_true',
    #                     help='Do lower case on input')
    parser.add_argument('--input_suffix', type=str, default='v4_gold_conll')
    parser.add_argument('--input_format', type=str, default='conll-2012',
                        choices=["conll-2012", "semeval-2010"],
                        help='Input format conll-2012 (default) or semeval-2010')

    args = parser.parse_args()
    logger.info(args)
    os.makedirs(args.output_dir, exist_ok=True)

    minimize_language(args)
