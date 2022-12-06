from dataclasses import dataclass
from typing import Optional

import mlflow
import torch
from torch import nn
from transformers import ElectraModel, ElectraPreTrainedModel, ElectraTokenizer

from event_classify.config import Config, Output
from event_classify.datasets import EventClassificationLabels
from event_classify.label_smoothing import LabelSmoothingLoss

EVENT_CLASS_WEIGHTS = torch.tensor([1 / 0.35, 1 / 0.009, 1 / 0.405, 1 / 0.23])
EVENT_PROPERTIES = {
    "categories": 4,
    "iterative": 1,
    "character_speech": 3,
    "thought_representation": 1,
    "mental": 1,
}


class MultiLossLayer(nn.Module):
    def __init__(self, length):
        """
        Multi loss with learned parameters

        The values for sigma are learned and give the loss of uncertain labels more weight.
        """
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.ones(length) * 0.5)

    def forward(self, losses: torch.Tensor):
        loss_scalers = 1 / (2 * torch.exp(self.log_sigmas))
        for i, loss_scaler in enumerate(loss_scalers):
            mlflow.log_metric(f"Scale loss {i}", float(loss_scalers[i].item()))
        return torch.mean(loss_scalers * losses + self.log_sigmas)


@dataclass
class EventClassificationOutput:
    event_type: torch.Tensor
    iterative: torch.Tensor
    speech_type: torch.Tensor
    thought_representation: torch.Tensor
    mental: torch.Tensor
    loss: Optional[torch.Tensor] = None


class ClassificationHead(nn.Module):
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraForEventClassification(ElectraPreTrainedModel):
    def __init__(self, config, event_config: Config):
        super().__init__(config)
        self.event_config = event_config
        if event_config.label_smoothing:
            self.event_type_critereon = LabelSmoothingLoss(weight=EVENT_CLASS_WEIGHTS)
        else:
            self.event_type_critereon = nn.CrossEntropyLoss(weight=EVENT_CLASS_WEIGHTS)
        self.property_loss = nn.CrossEntropyLoss()
        self.electra = ElectraModel(config)
        self.config = config
        self.event_type = ClassificationHead(
            config, num_labels=EVENT_PROPERTIES["categories"]
        )
        self.iterative = ClassificationHead(
            config, num_labels=EVENT_PROPERTIES["iterative"]
        )
        if (
            self.event_config.dynamic_loss_weighting
            and len(self.event_config.optimize_outputs) > 1
        ):
            self.multi_loss = MultiLossLayer(len(self.event_config.optimize_outputs))
        else:
            self.loss_weights = nn.parameter.Parameter(
                torch.tensor(self.event_config.static_loss_weights), requires_grad=False
            )

        self.thought_embedding = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.character_speech = nn.Sequential(
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, EVENT_PROPERTIES["character_speech"]),
        )
        self.thought_representation = nn.Sequential(
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, EVENT_PROPERTIES["thought_representation"]),
        )
        self.mental = nn.Sequential(
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, EVENT_PROPERTIES["mental"]),
        )
        self.thought_representation_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(97 / 3)
        )
        self.mental_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(77 / 23))
        self.iterative_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(83 / 17)
        )
        self.speech_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1 / 0.52, 1 / 0.02, 1 / 0.45])
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels: Optional[EventClassificationLabels] = None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
        )

        sequence_output = discriminator_hidden_states[0]
        logits_kind = self.event_type(sequence_output)
        logits_iterative = self.iterative(sequence_output)
        thought_embedding = self.thought_embedding(sequence_output[:, 0, :])
        logits_speech = self.character_speech(thought_embedding)
        logits_thought_representation = self.thought_representation(thought_embedding)
        logits_mental = self.mental(thought_embedding)

        loss = None
        if labels is not None:
            losses = []
            if Output.EVENT_KIND in self.event_config.optimize_outputs:
                losses.append(self.event_type_critereon(logits_kind, labels.event_type))
            if Output.THOUGHT_REPRESENTATION in self.event_config.optimize_outputs:
                losses.append(
                    self.thought_representation_criterion(
                        logits_thought_representation.squeeze(),
                        labels.thought_representation,
                    )
                )
            if Output.SPEECH in self.event_config.optimize_outputs:
                losses.append(self.speech_criterion(logits_speech, labels.speech_type))
            # only for all events that are not non events
            mental_defined = torch.masked_select(
                logits_mental.squeeze(), labels.event_type != 0
            )
            iterative_defined = torch.masked_select(
                logits_iterative.squeeze(), labels.event_type != 0
            )
            if len(mental_defined) > 0:
                if Output.MENTAL in self.event_config.optimize_outputs:
                    losses.append(self.mental_criterion(mental_defined, labels.mental))
                if Output.ITERATIVE in self.event_config.optimize_outputs:
                    losses.append(
                        self.iterative_criterion(iterative_defined, labels.iterative)
                    )
            else:
                if Output.MENTAL in self.event_config.optimize_outputs:
                    losses.append(torch.tensor(0).to(self.device))
                if Output.ITERATIVE in self.event_config.optimize_outputs:
                    losses.append(torch.tensor(0).to(self.device))
            if (
                self.event_config.dynamic_loss_weighting
                and len(self.event_config.optimize_outputs) > 1
            ):
                loss = self.multi_loss(torch.stack(losses))
            else:
                loss = torch.mean(torch.stack(losses) * self.loss_weights)

        return EventClassificationOutput(
            loss=loss,
            event_type=torch.argmax(logits_kind, 1),
            iterative=torch.argmax(logits_iterative, 1),
            speech_type=torch.argmax(logits_speech, 1),
            thought_representation=torch.argmax(logits_speech, 1),
            mental=torch.argmax(logits_mental, 1),
        )
