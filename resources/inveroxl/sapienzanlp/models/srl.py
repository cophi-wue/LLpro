import logging
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import transformers as tr

from sapienzanlp.common.logging import get_logger
from sapienzanlp.data.labels import Labels
from sapienzanlp.models.layers import (
    StateEncoder,
    WordEncoder,
    FullyConnectedLSTM,
)
from sapienzanlp.models.model import Model

logger = get_logger(level=logging.DEBUG)


@dataclass
class SrlOutput(tr.file_utils.ModelOutput):
    predicates: Optional[torch.Tensor] = None
    senses: Optional[torch.Tensor] = None
    arguments: Optional[torch.Tensor] = None


class CrosslingualSrl(Model):
    def __init__(
        self,
        labels: Dict[str, Labels],
        predictor: str,
        language_model: str,
        language_model_fine_tuning: bool = False,
        word_emb_size: int = 512,
        word_emb_dropout: float = 0.1,
        predicate_identification_size: int = 32,
        predicate_identification_dropout: float = 0.1,
        predicate_disambiguation_size: int = 512,
        predicate_disambiguation_dropout: float = 0.1,
        argument_aware_size: int = 256,
        argument_aware_dropout: float = 0.1,
        predicate_specific_size: int = 512,
        predicate_specific_dropout: float = 0.1,
        argument_specific_size: int = 512,
        argument_specific_dropout: float = 0.1,
        word_sequence_encoder_hidden_size: int = 512,
        word_sequence_encoder_layers: int = 1,
        word_sequence_encoder_dropout: float = 0.1,
        argument_sequence_encoder_hidden_size: int = 512,
        argument_sequence_encoder_layers: int = 1,
        argument_sequence_encoder_dropout: float = 0.1,
        device: str = None,
    ):
        super(CrosslingualSrl, self).__init__(labels, device, predictor)
        # params
        self.language_model = language_model
        self.inventories = ["ca", "cz", "de", "en", "es", "va", "zh"]
        # layers
        # word encoder
        self.word_encoder = WordEncoder(
            language_model=language_model,
            fine_tune=language_model_fine_tuning,
            output_layer="concat",
            dropout=word_emb_dropout,
            output_size=word_emb_size,
        )
        self.sequence_encoder = FullyConnectedLSTM(
            input_size=word_emb_size,
            hidden_size=word_sequence_encoder_hidden_size,
            num_layers=word_sequence_encoder_layers,
            dropout=word_sequence_encoder_dropout,
        )

        # SRL Task specific layers
        sequence_encoder_output = self.sequence_encoder.output_size

        # Predicate identification
        self.predicate_encoder = StateEncoder(
            input_size=sequence_encoder_output,
            state_size=predicate_identification_size,
            dropout=predicate_identification_dropout,
        )

        self.predicate_scorer = nn.ModuleDict()
        for inventory in self.labels:
            # if inventory == "cs":
            #     self.predicate_scorer["cz"] = nn.Linear(self.predicate_encoder.output_size, 2)
            # else:
            self.predicate_scorer[inventory] = nn.Linear(self.predicate_encoder.output_size, 2)

        # Predicate disambiguation
        self.sense_encoder = StateEncoder(
            input_size=sequence_encoder_output,
            state_size=predicate_disambiguation_size,
            dropout=predicate_disambiguation_dropout,
        )

        self.sense_scorer = nn.ModuleDict()
        for inventory in self.labels:
            # if inventory == "cs":
            #     self.sense_scorer["cz"] = nn.Linear(
            #         self.sense_encoder.output_size, labels[inventory].get_label_size("senses")
            #     )
            # else:
            self.sense_scorer[inventory] = nn.Linear(
                self.sense_encoder.output_size, labels[inventory].get_label_size("senses")
            )

        # Argument identification/disambiguation
        self.predicate_specific_encoder = StateEncoder(
            input_size=sequence_encoder_output,
            state_size=predicate_specific_size,
            dropout=predicate_specific_dropout,
        )
        self.argument_specific_encoder = StateEncoder(
            input_size=sequence_encoder_output,
            state_size=argument_specific_size,
            dropout=argument_specific_dropout,
        )
        self.argument_sequence_encoder = FullyConnectedLSTM(
            input_size=predicate_specific_size + argument_specific_size,
            hidden_size=argument_sequence_encoder_hidden_size,
            num_layers=argument_sequence_encoder_layers,
            dropout=argument_sequence_encoder_dropout,
        )
        self.argument_aware_encoder = StateEncoder(
            input_size=self.argument_sequence_encoder.output_size,
            state_size=argument_aware_size,
            dropout=argument_aware_dropout,
        )

        self.role_scorer = nn.ModuleDict()
        for inventory in self.labels:
            # if inventory == "cs":
            #     self.role_scorer["cz"] = nn.Linear(
            #         self.argument_aware_encoder.output_size, labels[inventory].get_label_size("roles")
            #     )
            # else:
            self.role_scorer[inventory] = nn.Linear(
                self.argument_aware_encoder.output_size, labels[inventory].get_label_size("roles")
            )

    def forward(
        self,
        # language: str = None,
        input_ids: torch.Tensor = None,
        offsets: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        sentence_lengths: torch.Tensor = None,
        predicate_indices: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        predict_predicates = bool(predicate_indices is None)
        # word embeddings
        embeddings = self.word_encoder(input_ids, attention_mask, token_type_ids, offsets)
        # encodings from the sequences
        sequence_states = self.sequence_encoder(embeddings, sentence_lengths)
        # predicate identification
        predicate_encodings = self.predicate_encoder(sequence_states)
        outputs = {}
        for inventory in self.inventories:
            predicate_scores = self.predicate_scorer[inventory](predicate_encodings)
            if predict_predicates:
                # get predicates from classifier if not provided in input
                predicate_indices = torch.argmax(predicate_scores, dim=-1)
                if 1 not in predicate_indices:
                    # check without predicates
                    # this kinda sucks, need help :(
                    outputs[inventory] = SrlOutput(predicates=predicate_scores)
                    continue

            # predicate disambiguation
            sense_encodings = self.sense_encoder(sequence_states)
            sense_encodings = sense_encodings[predicate_indices == 1]
            sense_scores = self.sense_scorer[inventory](sense_encodings)
            # argument identification/disambiguation
            # get batch sequences length
            timesteps = sequence_states.shape[1]
            # predicate states encodings
            pred_spec_encodings = self.predicate_specific_encoder(sequence_states)
            pred_spec_encodings = pred_spec_encodings.unsqueeze(2)
            pred_spec_encodings = pred_spec_encodings.expand(-1, -1, timesteps, -1)
            # argument states encodings
            arg_spec_encodings = self.argument_specific_encoder(sequence_states)
            arg_spec_encodings = arg_spec_encodings.unsqueeze(1)
            arg_spec_encodings = arg_spec_encodings.expand(-1, timesteps, -1, -1)
            # predicate/argument aware encodings
            predicate_argument_states = torch.cat((pred_spec_encodings, arg_spec_encodings), dim=-1)
            # get predicate vectors
            predicate_argument_states = predicate_argument_states[predicate_indices == 1]
            # get row lengths and repeat by the number of predicates in the row
            predicates_counter = torch.sum(predicate_indices, dim=1)
            argument_sequence_lengths = torch.repeat_interleave(
                sentence_lengths, predicates_counter
            )
            max_argument_sequence_length = torch.max(argument_sequence_lengths)
            predicate_argument_states = predicate_argument_states[
                :, :max_argument_sequence_length, :
            ]
            argument_encodings = self.argument_sequence_encoder(
                predicate_argument_states, argument_sequence_lengths
            )
            # arguments encoding
            argument_encodings = self.argument_aware_encoder(argument_encodings)
            role_scores = self.role_scorer[inventory](argument_encodings)

            outputs[inventory] = SrlOutput(
                predicates=predicate_scores, senses=sense_scores, arguments=role_scores
            )
        return outputs
