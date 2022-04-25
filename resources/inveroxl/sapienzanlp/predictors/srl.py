import logging
import math
from typing import Union, List, Optional, Tuple, Any, Dict

import torch
from overrides import overrides

from sapienzanlp.common.logging import get_logger
from sapienzanlp.data.model_io.model_inputs import ModelInputs
from sapienzanlp.data.model_io.sentence import SrlSentence, Sentence
from sapienzanlp.data.model_io.word import Word, Predicate, Argument
from sapienzanlp.models.model import Model
from sapienzanlp.predictors.predictor import Predictor
from sapienzanlp.preprocessing.spacy_tokenizer import SpacyTokenizer
from sapienzanlp.preprocessing.transformers_processor import TransformersProcessor

logger = get_logger(level=logging.DEBUG)


class SemanticRoleLabeler(Predictor):
    """
    Semantic Role Labeling predictor class.

    Args:
        model (:obj:`~sapienzanlp.models.Model`):
            The :obj:`~sapienzanlp.models.Model` that this predictor will use to produce the output.
        language (:obj:`str`):
            Language of the text in input to the predictor. It is used by the
            :obj:`~sapienzanlp.preprocessing.Tokenizer` to preprocess the text. It is ignored if the text is
            already preprocessed.

    """

    def __init__(self, model: Model, language: str = "en", **kwargs):
        super(SemanticRoleLabeler, self).__init__(model, language)
        # tokenizer args
        self.tokenizer_kwargs["split_on_spaces"] = kwargs.get("split_on_spaces", False)
        # lazy loading of the tokenizer
        # it will be loaded if the input to `__call__` is not tokenized
        self.tokenizer = SpacyTokenizer
        self.processor = TransformersProcessor(model.word_encoder.language_model)
        self.processor.add_to_tensor_inputs("sentence_lengths")
        self.labels = self.model.labels

    def __call__(
        self,
        text: Union[str, List[str], List[List[str]], List[Word], List[List[Word]]],
        is_split_into_words: bool = False,
        *args,
        **kwargs,
    ) -> Dict[Any, List[Sentence]]:
        """
        Expose the inference to the user. It tags the the input text with semantic roles for each of the
        predicates the model find in it.

        Args:
            text (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`, :obj:`List[Word]`, :obj:`List[List[Word]]`):
                Text to tag. It can be a single string, a batch of string and pre-tokenized strings.
            is_split_into_words (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True` and the input is a string, the input is split on spaces.

        Returns:
            :obj:`List[Sentence]`: The input batch tagged with labels from the model.

        """
        text, model_inputs = self.prepare_input_for_model(
            text, is_split_into_words, *args, **kwargs
        )
        model_outputs = self.model(**model_inputs)
        model_outputs = self.decode(model_inputs, model_outputs)
        return self.make_output(text, model_outputs)

    @overrides
    def prepare_input_for_model(
        self,
        text: Union[str, List[str], List[List[str]], List[Word], List[List[Word]]],
        is_split_into_words: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[List[List[Word]], ModelInputs]:
        """
        Prepare the input for the model.

        Args:
            text (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`, :obj:`List[Word]`, :obj:`List[List[Word]]`):
                Text to tag. It can be a single string, a batch of string and pre-tokenized strings.
            is_split_into_words (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True` and the input is a string, the input is split on spaces.

        Returns:
            :obj:`List[List[Word]]`, :obj:`ModelInputs`: The input processed ready for the model.
        """
        text = self.preprocess_text(text, is_split_into_words)
        # process the input for the model
        model_inputs = self.processor(text, **kwargs)
        # move to the correct device (GPU or CPU)
        model_inputs = model_inputs.to(device=self.device)
        return text, model_inputs

    def make_output(
        self, texts: List[List[Word]], model_outputs: Any
    ) -> Dict[Any, List[SrlSentence]]:
        """
        Using the input and the output of the model, builds the model output in a readable way.

        Args:
            texts (:obj:`List[List[Word]]`):
                Input to tag preprocessed.
            model_outputs (:obj:`Any`):
                The output from the model.

        Returns:
            :obj:`List[SrlSentence]`: The input batch tagged with labels from the model.
        """
        output_lang_sentences = {}
        for inventory, outputs in model_outputs.items():
            predicates, senses, roles = outputs
            output_sentences = []
            for sid, (text, sense) in enumerate(zip(texts, senses)):
                sentence = SrlSentence()
                for word_index, (w, s) in enumerate(zip(text, sense)):
                    if s != "_" and s != "lemma":
                        w = Predicate(w.text, w.index, sense=s)
                        sentence.predicates.append(w)
                    sentence.append(w)
                if sid in roles:
                    for predicate, bio_roles in zip(sentence.predicates, roles[sid].values()):
                        span_roles = self.bio_to_spans(bio_roles)
                        predicate.arguments = [
                            Argument(r, predicate, [sentence[start:end]], start, end)
                            for r, start, end in span_roles
                        ]
                output_sentences.append(sentence)
            output_lang_sentences[inventory] = output_sentences
        return output_lang_sentences

    @overrides
    def decode(
        self, model_inputs: ModelInputs, model_outputs: Any
    ) -> Dict[Any, Tuple[List[List[int]], List[List[str]], Any]]:
        """
        Decoding function for the Semantic Role Labeling models.

        Args:
            model_inputs (:obj:`ModelInputs`):
                Inputs to the model.
            model_outputs (:obj:`Any`):
                Output from the model.

        Returns:
            :obj:`List[List[int]]`, :obj:`List[List[str]]`, :obj:`Dict`:
            The decoded model output (predicate, senses and roles).
        """
        decode_outputs = {}
        for inventory, outputs in model_outputs.items():
            # predicate identification
            predicates = self._decode_predicates(
                model_inputs.sentence_lengths, outputs.predicates
            )
            # prepare identified predicates for senses and roles
            # this new data structure has the following form
            # [[sentence_index, predicate_index], [..., ...], ...]
            predicates_ = []
            for sentence_index, predicate in enumerate(predicates):
                predicates_ += [
                    (sentence_index, pred_index)
                    for pred_index, is_pred in enumerate(predicate)
                    if is_pred == 1
                ]
            # predicate disambiguation
            senses = self._decode_senses(
                model_inputs.sentence_lengths, predicates_, outputs.senses, inventory
            )
            # argument identification/disambiguation
            roles = self._decoding_roles(
                model_inputs.sentence_lengths, predicates_, outputs.arguments, inventory
            )
            decode_outputs[inventory] = (predicates, senses, roles)
        # returns the result
        return decode_outputs

    def _decode_predicates(
        self, sentence_lengths: List[int], predicate_scores: torch.Tensor
    ) -> List[List[int]]:
        """
        Return :obj:`1` or :obj:`0` for every token in the bach.
        :obj:`1` if it is a predicate, :obj:`0` otherwise.

        Args:
            sentence_lengths (:obj:`List[int]`):
                A list of the length of the sentences in the batch.
            predicate_scores (:obj:`torch.Tensor`):
                The output of the model about predicate identification.

        Returns:
            :obj:`List[List[int]]`:
                An identification matrix containing :obj:`1` if a token is a predicate, :obj:`0` otherwise.

        """
        predicate_indices = torch.argmax(predicate_scores, dim=-1).tolist()
        outputs = []
        for predicates, sentence_length in zip(predicate_indices, sentence_lengths):
            # remove CLS, SEP and padding from predictions
            outputs.append(predicates[1 : sentence_length - 1])
        return outputs

    def _decode_senses(
        self,
        sentence_lengths: List[int],
        predicates: List[List[int]],
        sense_scores: torch.Tensor,
        inventory: str,
    ) -> List[List[str]]:
        """
        Return the label for every predicate in the sentences.

        Args:
            sentence_lengths (:obj:`List[int]`):
                A list of the length of the sentences in the batch.
           predicates (:obj:`List[List[int]]`):
                An indicator matrix that tells which token is a predicate.
            sense_scores (:obj:`torch.Tensor`):
                The output of the model about predicate labels.

        Returns:
            :obj:`List[List[str]]`:
                The labels for each word in the batch.
        """
        # placeholder list. `sentence_length - 2` because it includes CLS and SEP
        senses = [
            [self.labels[inventory].get_label_from_index(0, "senses")] * (sentence_length - 2)
            for sentence_length in sentence_lengths
        ]
        # TODO if there are no predicates in the sentence, the sense scores are None
        # need to find a different method probably
        if sense_scores is None:
            return senses
        sense_ids = torch.argmax(sense_scores, dim=-1).tolist()
        # this for is needed since the output matrix is a flat vector of all the
        # predicates in the batch.
        for (sentence_index, predicate_index), sense_id in zip(predicates, sense_ids):
            senses[sentence_index][predicate_index] = self.labels[inventory].get_label_from_index(
                sense_id, "senses"
            )
        return senses

    def _decoding_roles(
        self,
        sentence_lengths: List[int],
        predicates: List[List[int]],
        role_scores: torch.Tensor,
        inventory: str,
    ) -> Any:
        """
        Return the label for every argument in the sentences.

        Args:
            sentence_lengths (:obj:`List[int]`):
                A list of the length of the sentences in the batch.
            predicates (:obj:`List[List[int]]`):
                An indicator matrix that tells which token is a predicate.
            role_scores (:obj:`torch.Tensor`):
                The output of the model about argument labels.

        Returns:

        """
        raise NotImplementedError

    @staticmethod
    def bio_to_spans(tags: List[str]) -> List[List[Union[int, Any]]]:
        """
        Convert BIO label to a sequence of labeled arguments.

        Args:
            tags:

        Returns:

        """
        span = []

        for (i, tag) in enumerate(tags):
            if tag == "_" or "-V" in tag:
                continue
            label = tag[2:]
            if tag[0] == "B" or len(span) == 0 or label != tags[i - 1][2:]:
                span.append([label, i, -1])
            # Close current span.
            if i == len(tags) - 1 or tags[i + 1][0] == "B" or label != tags[i + 1][2:]:
                span[-1][2] = i + 1

        return span


class DependencySemanticRoleLabeler(SemanticRoleLabeler):
    """Dependency based Semantic Role Labeling predictor."""

    def __init__(self, model: Model, language: str = "en", **kwargs):
        super(DependencySemanticRoleLabeler, self).__init__(model, language, **kwargs)

    @overrides
    def _decoding_roles(
        self,
        sentence_lengths: List[int],
        predicates: List[List[int]],
        role_scores: torch.Tensor,
        inventory: str,
    ) -> Dict:
        """
        Return the label for every argument in the sentences.

        Args:
            sentence_lengths (:obj:`List[int]`):
                A list of the length of the sentences in the batch.
            predicates (:obj:`List[List[int]]`):
                An indicator matrix that tells which token is a predicate.
            role_scores (:obj:`torch.Tensor`):
                The output of the model about argument labels.

        Returns:
            :obj:`Dict`: A dictionary in which there is the set of roles for each predicate in the batch.
        """
        roles = {i: {} for i in range(len(predicates))}

        if role_scores is None:
            # TODO if there are no predicates in the sentence, the role scores are None
            # need to find a different method probably
            return roles

        role_ids = torch.argmax(role_scores, dim=-1).tolist()
        for (sid, predicate_index), predicate_role_ids in zip(predicates, role_ids):
            sentence_length = sentence_lengths[sid]
            predicate_role_ids = predicate_role_ids[: sentence_length - 2]
            # all labels are BIO-tags for consistency with span-based
            predicate_roles = [
                self.labels[inventory].get_label_from_index(r, "roles") for r in predicate_role_ids
            ]
            roles[sid][predicate_index] = predicate_roles

        return roles


class SpanSemanticRoleLabeler(SemanticRoleLabeler):
    """Span based Semantic Role Labeling predictor."""

    def __init__(self, model: Model, language: str = "en", **kwargs):
        super(SpanSemanticRoleLabeler, self).__init__(model, language, **kwargs)
        self.role_transition_matrix = self.build_transition_matrix()
        self.role_start_transitions = self.build_start_transition_matrix()

    @overrides
    def _decoding_roles(
        self,
        sentence_lengths: List[int],
        predicates: List[List[int]],
        role_scores: torch.Tensor,
        inventory: str,
    ) -> Dict:
        """
        Return the label for every argument in the sentences.

        Args:
            sentence_lengths (:obj:`List[int]`):
                A list of the length of the sentences in the batch.
            predicates (:obj:`List[List[int]]`):
                An indicator matrix that tells which token is a predicate.
            role_scores (:obj:`torch.Tensor`):
                The output of the model about argument labels.

        Returns:
            :obj:`Dict`: A dictionary in which there is the set of roles for each predicate in the batch.

        """
        roles = {i: {} for i in range(len(predicates))}

        if role_scores is None:
            # TODO if there are no predicates in the sentence, the role scores are None
            # need to find a different method probably
            return roles

        # role_emissions = torch.argmax(role_scores, dim=-1)
        for (sid, predicate_index), predicate_role_emissions in zip(predicates, role_scores):
            sentence_length = sentence_lengths[sid]
            predicate_role_emissions = predicate_role_emissions[: sentence_length - 2]
            predicate_role_ids, _ = self.viterbi_decode(
                predicate_role_emissions.to("cpu"),
                torch.as_tensor(self.role_transition_matrix[inventory]),
                allowed_start_transitions=torch.as_tensor(self.role_start_transitions[inventory]),
            )
            predicate_roles = [
                self.labels[inventory].get_label_from_index(r, "roles") for r in predicate_role_ids
            ]
            roles[sid][predicate_index] = predicate_roles
        return roles

    def build_transition_matrix(self):
        """

        Returns:

        """
        role_transition_matrices = {}

        for language, language_labels in self.labels.items():
            role_transition_matrix = []
            for i in range(language_labels.get_label_size("roles")):
                previous_label = language_labels.get_label_from_index(i, "roles")
                role_transitions = []
                for j in range(language_labels.get_label_size("roles")):
                    label = language_labels.get_label_from_index(j, "roles")
                    if i != j and label[0] == "I" and not previous_label == "B" + label[1:]:
                        role_transitions.append(float("-inf"))
                    else:
                        role_transitions.append(0.0)
                role_transition_matrix.append(role_transitions)
            role_transition_matrix = torch.as_tensor(role_transition_matrix)
            role_transition_matrices[language] = role_transition_matrix

        return role_transition_matrices

    def build_start_transition_matrix(self):
        """

        Returns:

        """
        role_start_transitions = {}

        for language, language_labels in self.labels.items():
            language_rst = []
            for i in range(language_labels.get_label_size("roles")):
                label = language_labels.get_label_from_index(i, "roles")
                if label[0] == "I":
                    language_rst.append(float("-inf"))
                else:
                    language_rst.append(0.0)
            language_rst = torch.as_tensor(language_rst)
            role_start_transitions[language] = language_rst

        return role_start_transitions

    @staticmethod
    def viterbi_decode(
        tag_sequence: torch.Tensor,
        transition_matrix: torch.Tensor,
        tag_observations: Optional[List[int]] = None,
        allowed_start_transitions: torch.Tensor = None,
        allowed_end_transitions: torch.Tensor = None,
        top_k: int = None,
    ):
        """
        Perform Viterbi decoding in log space over a sequence given a transition matrix
        specifying pairwise (transition) potentials between tags and a matrix of shape
        (sequence_length, num_tags) specifying unary potentials for possible tags per
        timestep.

        Args:
            tag_sequence : `torch.Tensor`, required.
                A tensor of shape (sequence_length, num_tags) representing scores for
                a set of tags over a given sequence.
            transition_matrix : `torch.Tensor`, required.
                A tensor of shape (num_tags, num_tags) representing the binary potentials
                for transitioning between a given pair of tags.
            tag_observations : `Optional[List[int]]`, optional, (default = `None`)
                A list of length `sequence_length` containing the class ids of observed
                elements in the sequence, with unobserved elements being set to -1. Note that
                it is possible to provide evidence which results in degenerate labelings if
                the sequences of tags you provide as evidence cannot transition between each
                other, or those transitions are extremely unlikely. In this situation we log a
                warning, but the responsibility for providing self-consistent evidence ultimately
                lies with the user.
            allowed_start_transitions : `torch.Tensor`, optional, (default = `None`)
                An optional tensor of shape (num_tags,) describing which tags the START token
                may transition *to*. If provided, additional transition constraints will be used for
                determining the start element of the sequence.
            allowed_end_transitions : `torch.Tensor`, optional, (default = `None`)
                An optional tensor of shape (num_tags,) describing which tags may transition *to* the
                end tag. If provided, additional transition constraints will be used for determining
                the end element of the sequence.
            top_k : `int`, optional, (default = `None`)
                Optional integer specifying how many of the top paths to return. For top_k>=1, returns
                a tuple of two lists: top_k_paths, top_k_scores, For top_k==None, returns a flattened
                tuple with just the top path and its score (not in lists, for backwards compatibility).
        # Returns
        viterbi_path : `List[int]`
            The tag indices of the maximum likelihood tag sequence.
        viterbi_score : `torch.Tensor`
            The score of the viterbi path.
        """
        if top_k is None:
            top_k = 1
            flatten_output = True
        elif top_k >= 1:
            flatten_output = False
        else:
            raise ValueError(
                f"top_k must be either None or an integer >=1. Instead received {top_k}"
            )

        sequence_length, num_tags = list(tag_sequence.size())

        has_start_end_restrictions = (
            allowed_end_transitions is not None or allowed_start_transitions is not None
        )

        if has_start_end_restrictions:

            if allowed_end_transitions is None:
                allowed_end_transitions = torch.zeros(num_tags)
            if allowed_start_transitions is None:
                allowed_start_transitions = torch.zeros(num_tags)

            num_tags = num_tags + 2
            new_transition_matrix = torch.zeros(num_tags, num_tags)
            new_transition_matrix[:-2, :-2] = transition_matrix

            # Start and end transitions are fully defined, but cannot transition between each other.

            allowed_start_transitions = torch.cat(
                [allowed_start_transitions, torch.tensor([-math.inf, -math.inf])]
            )
            allowed_end_transitions = torch.cat(
                [allowed_end_transitions, torch.tensor([-math.inf, -math.inf])]
            )

            # First define how we may transition FROM the start and end tags.
            new_transition_matrix[-2, :] = allowed_start_transitions
            # We cannot transition from the end tag to any tag.
            new_transition_matrix[-1, :] = -math.inf

            new_transition_matrix[:, -1] = allowed_end_transitions
            # We cannot transition to the start tag from any tag.
            new_transition_matrix[:, -2] = -math.inf

            transition_matrix = new_transition_matrix

        if tag_observations:
            if len(tag_observations) != sequence_length:
                raise ValueError(
                    "Observations were provided, but they were not the same length "
                    "as the sequence. Found sequence of length: {} and evidence: {}".format(
                        sequence_length, tag_observations
                    )
                )
        else:
            tag_observations = [-1 for _ in range(sequence_length)]

        if has_start_end_restrictions:
            tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
            zero_sentinel = torch.zeros(1, num_tags)
            extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
            tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
            tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
            sequence_length = tag_sequence.size(0)

        path_scores = []
        path_indices = []

        if tag_observations[0] != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[tag_observations[0]] = 100000.0
            path_scores.append(one_hot.unsqueeze(0))
        else:
            path_scores.append(tag_sequence[0, :].unsqueeze(0))

        # Evaluate the scores for all possible paths.
        for timestep in range(1, sequence_length):
            # Add pairwise potentials to current scores.
            summed_potentials = path_scores[timestep - 1].unsqueeze(2) + transition_matrix
            summed_potentials = summed_potentials.view(-1, num_tags)

            # Best pairwise potential path score from the previous timestep.
            max_k = min(summed_potentials.size()[0], top_k)
            scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)

            # If we have an observation for this timestep, use it
            # instead of the distribution over tags.
            observation = tag_observations[timestep]
            # Warn the user if they have passed
            # invalid/extremely unlikely evidence.
            if tag_observations[timestep - 1] != -1 and observation != -1:
                if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                    logger.warning(
                        "The pairwise potential between tags you have passed as "
                        "observations is extremely unlikely. Double check your evidence "
                        "or transition potentials!"
                    )
            if observation != -1:
                one_hot = torch.zeros(num_tags)
                one_hot[observation] = 100000.0
                path_scores.append(one_hot.unsqueeze(0))
            else:
                path_scores.append(tag_sequence[timestep, :] + scores)
            path_indices.append(paths.squeeze())

        # Construct the most likely sequence backwards.
        path_scores_v = path_scores[-1].view(-1)
        max_k = min(path_scores_v.size()[0], top_k)
        viterbi_scores, best_paths = torch.topk(path_scores_v, k=max_k, dim=0)
        viterbi_paths = []
        for i in range(max_k):
            viterbi_path = [best_paths[i]]
            for backward_timestep in reversed(path_indices):
                viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
            # Reverse the backward path.
            viterbi_path.reverse()

            if has_start_end_restrictions:
                viterbi_path = viterbi_path[1:-1]

            # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
            viterbi_path = [j % num_tags for j in viterbi_path]
            viterbi_paths.append(viterbi_path)

        if flatten_output:
            return viterbi_paths[0], viterbi_scores[0]

        return viterbi_paths, viterbi_scores
