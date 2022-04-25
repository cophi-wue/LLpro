from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
import transformers as tr
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sapienzanlp.common.from_config import FromConfig


@dataclass
class TransformersEmbedderOutput(tr.file_utils.ModelOutput):
    """Class for model's outputs."""

    embeddings: torch.Tensor = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class TransformersEmbedder(nn.Module, FromConfig):
    """
    Transformer Embedder class.
    Word level embeddings from various transformer architectures from Huggingface Trasnformers API.

    Args:
        model (:obj:`str`, :obj:`tr.PreTrainedModel`):
            Transformer model to use (https://huggingface.co/models).
        return_words (:obj:`bool`, optional, defaults to :obj:`True`):
            If ``True`` it returns the word-level embeddings by computing the mean of the
            sub-words embeddings.
        output_layer (:obj:`str`, optional, defaults to :obj:`last`):
            What output to get from the transformer model. The last hidden state (``last``),
            the concatenation of the last four hidden layers (``concat``), the sum of the last four hidden
            layers (``sum``), the pooled output (``pooled``).
        fine_tune (:obj:`bool`, optional, defaults to :obj:`True`):
            If ``True``, the transformer model is fine-tuned during training.
        return_all (:obj:`bool`, optional, defaults to :obj:`False`):
            If ``True``, returns all the outputs from the HuggingFace model.
    """

    def __init__(
        self,
        model: Union[str, tr.PreTrainedModel],
        return_words: bool = True,
        output_layer: str = "last",
        fine_tune: bool = True,
        return_all: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(model, str):
            config = tr.AutoConfig.from_pretrained(
                model, output_hidden_states=True, output_attention=True
            )
            self.transformer_model = tr.AutoModel.from_config(config)
        else:
            self.transformer_model = model
        self.return_words = return_words
        self.output_layer = output_layer
        self.return_all = return_all
        if not fine_tune:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        offsets: torch.LongTensor = None,
        *args,
        **kwargs,
    ) -> TransformersEmbedderOutput:
        """
        Forward method of the PyTorch module.

        Args:
            input_ids (:obj:`torch.Tensor`, optional):
                Input ids for the transformer model.
            attention_mask (:obj:`torch.Tensor`, optional):
                Attention mask for the transformer model.
            token_type_ids (:obj:`torch.Tensor`, optional):
                Token type ids for the transformer model.
            offsets (:obj:`torch.Tensor`, optional):
                Offsets of the sub-token, used to reconstruct the word embeddings.

        Returns:
             :obj:`TransformersEmbedderOutput`:
                Word level embeddings plus the output of the transformer model.
        """
        # Some of the HuggingFace models don't have the
        # token_type_ids parameter and fail even when it's given as None.
        max_type_id = token_type_ids.max()
        if max_type_id == 0:
            token_type_ids = None
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        # Shape: [batch_size, num_subtoken, embedding_size].
        transformer_outputs = self.transformer_model(**inputs)
        if self.output_layer == "last":
            embeddings = transformer_outputs.last_hidden_state
        elif self.output_layer == "concat":
            embeddings = transformer_outputs.hidden_states[-4:]
            embeddings = torch.cat(embeddings, dim=-1)
        elif self.output_layer == "sum":
            embeddings = transformer_outputs.hidden_states[-4:]
            embeddings = torch.stack(embeddings, dim=0).sum(dim=0)
        elif self.output_layer == "pooled":
            embeddings = transformer_outputs.pooler_output
        else:
            raise ValueError(
                "output_layer parameter not valid, choose between `last`, `concat`, "
                f"`sum`, `pooled`. Current value `{self.output_layer}`"
            )

        if self.return_words and offsets is None:
            raise ValueError(
                f"`return_words` is `True` but `offsets` was not passed to the model. "
                f"Cannot compute word embeddings. To solve:\n"
                f"- Set `return_words` to `False` or"
                f"- Pass `offsets` to the model during forward."
            )
        if self.return_words:
            # Shape: [batch_size, num_token, embedding_size].
            embeddings = self.merge_subword(embeddings, offsets)
        if self.return_all:
            return TransformersEmbedderOutput(
                embeddings=embeddings,
                last_hidden_state=transformer_outputs.last_hidden_state,
                hidden_states=transformer_outputs.hidden_states,
                pooler_output=transformer_outputs.pooler_output,
                attentions=transformer_outputs.attentions,
            )
        return TransformersEmbedderOutput(embeddings=embeddings)

    def merge_subword(
        self,
        embeddings: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Minimal version of ``scatter_mean``, from `pytorch_scatter <https://github.com/rusty1s/pytorch_scatter/>`_
        library, that is compatible for ONNX but works only for our case. It is used to compute word level
        embeddings from the transformer output.

        Args:
            embeddings (:obj:`torch.Tensor`):
                The embeddings tensor.
            indices (:obj:`torch.Tensor`):
                The subword indices.

        Returns:
            :obj:`torch.Tensor`

        """
        out = self.scatter_sum(embeddings, indices)
        ones = torch.ones(indices.size(), dtype=embeddings.dtype, device=embeddings.device)
        count = self.scatter_sum(ones, indices)
        count.clamp_(1)
        count = self.broadcast(count, out)
        out.true_divide_(count)
        return out

    def scatter_sum(
        self,
        src: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Minimal version of ``scatter_sum``, from `pytorch_scatter <https://github.com/rusty1s/pytorch_scatter/>`_
        library, that is compatible for ONNX but works only for our case.

        Args:
            src (:obj:`torch.Tensor`):
                The source tensor.
            index (:obj:`torch.Tensor`):
                The indices of elements to scatter.

        Returns:
            :obj:`torch.Tensor`

        """
        index = self.broadcast(index, src)
        size = list(src.size())
        size[1] = index.max() + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(1, index, src)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> torch.nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if :obj:`new_num_tokens != config.vocab_size`.

        Args:
            new_num_tokens (:obj:`int`):
                The number of new tokens in the embedding matrix.

        Returns:
            :obj:`torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.

        """
        return self.transformer_model.resize_token_embeddings(new_num_tokens)

    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        Save a model and its configuration file to a directory.

        Args:
            save_directory (:obj:`str`, :obj:`Path`):
                Directory to which to save.

        Returns:

        """
        self.transformer_model.save_pretrained(save_directory)

    @staticmethod
    def broadcast(src: torch.Tensor, other: torch.Tensor):
        """
        Minimal version of ``broadcast``, from `pytorch_scatter <https://github.com/rusty1s/pytorch_scatter/>`_
        library, that is compatible for ONNX but works only for our case.

        Args:
            src (:obj:`torch.Tensor`):
                The source tensor.
            other (:obj:`torch.Tensor`):
                The tensor from which we want to broadcast.

        Returns:
            :obj:`torch.Tensor`

        """
        for _ in range(src.dim(), other.dim()):
            src = src.unsqueeze(-1)
        src = src.expand_as(other)
        return src

    @property
    def hidden_size(self) -> int:
        """
        Returns the hidden size of the transformer.

        Returns:
            :obj:`int`: Hidden size of ``self.transformer_model``.

        """
        multiplier = 4 if self.output_layer == "concat" else 1
        return self.transformer_model.config.hidden_size * multiplier


class WordEncoder(nn.Module, FromConfig):
    """
    A Word Encoder layer, that uses language models to produce word-level embeddings.

    Args:
        language_model (:obj:`str`, :obj:`tr.PreTrainedModel`):
            Transformer model to use (https://huggingface.co/models).
        return_words (:obj:`bool`, optional, defaults to :obj:`True`):
            If ``True`` it returns the word-level embeddings by computing the mean of the
            sub-words embeddings.
        output_layer (:obj:`str`, optional, defaults to :obj:`last`):
            What output to get from the transformer model. The last hidden state (``last``),
            the concatenation of the last four hidden layers (``concat``), the sum of the last four hidden
            layers (``sum``), the pooled output (``pooled``).
        dropout (:obj:`float`, optional, defaults to ``0.1``):
            Dropout hyper-parameter between layers.
        fine_tune (:obj:`bool`, optional, defaults to :obj:`True`):
            If ``True``, the transformer model is fine-tuned during training.
        output_size (:obj:`int`, optional):
            If provided, it reduces the output size of the transformer to this dimension.
        bias (:obj:`bool`, optional, defaults to :obj:`True`):
            The learnable bias of the module ``self.projection_layer``

    """

    def __init__(
        self,
        language_model: Union[str, tr.PreTrainedModel],
        return_words: bool = True,
        output_layer: str = "last",
        dropout: float = 0.1,
        fine_tune: bool = False,
        output_size: Optional[int] = None,
        bias: bool = True,
    ):
        super(WordEncoder, self).__init__()
        self.language_model = language_model
        self.transformer = TransformersEmbedder(
            language_model,
            return_words=return_words,
            output_layer=output_layer,
            fine_tune=fine_tune,
        )
        if not output_size:
            output_size = self.transformer.hidden_size
        # params
        self.output_size = output_size
        self.normalization_layer = nn.BatchNorm1d(self.transformer.hidden_size)
        self.projection_layer = nn.Linear(self.transformer.hidden_size, output_size, bias=bias)
        self.dropout_layer = nn.Dropout(dropout)
        self.activation_layer = Swish()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        offsets: torch.LongTensor = None,
        *args,
        **kwargs,
    ):
        """
        Forward method of the PyTorch module.

        Args:
            input_ids (:obj:`torch.Tensor`, optional):
                Input ids for the transformer model.
            attention_mask (:obj:`torch.Tensor`, optional):
                Attention mask for the transformer model.
            token_type_ids (:obj:`torch.Tensor`, optional):
                Token type ids for the transformer model.
            offsets (:obj:`torch.Tensor`, optional):
                Offsets of the sub-token, used to reconstruct the word embeddings.
        Returns:
             :obj:`TransformersEmbedderOutput`:
                Word level embeddings plus the output of the transformer model.

        """
        embeddings = self.transformer(input_ids, attention_mask, token_type_ids, offsets).embeddings
        if self.transformer.output_layer == "sum":
            embeddings *= 0.25
        embeddings = self.dropout_layer(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.normalization_layer(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.projection_layer(embeddings)
        embeddings = self.activation_layer(embeddings)
        embeddings = self.dropout_layer(embeddings)
        return embeddings


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class StateEncoder(nn.Module, FromConfig):
    """
    State encoder class.

    Args:
        input_size (:obj:`int`):
            The input size of the tensor.
        state_size (:obj:`int`):
        activation (:obj:`str`, optional, defaults to :obj:`swish`):
            Activation function to use after ``self.projection_layer``. Valid strings are:
                - ``identity``
                - ``relu``
                - ``swish``
        num_layers (:obj:`int`, optional, defaults to ``1``):
            Number of linear layers in the encoder.
        dropout (:obj:`float`, optional, defaults to ``0.1``):
            Dropout hyper-parameter between layers.

    """

    def __init__(
        self,
        input_size: int,
        state_size: int,
        activation: str = "swish",
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super(StateEncoder, self).__init__()
        # params
        self.output_size = state_size
        # layers
        linear_layers = [nn.Linear(input_size, state_size)]
        for _ in range(1, num_layers):
            linear_layers.append(nn.Linear(state_size, state_size))

        self.linear_layers = nn.ModuleList(linear_layers)

        if activation == "identity":
            self.activation = nn.Identity()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor):
        """
        Forward of the module.

        Args:
            input (:obj:`torch.Tensor`):
                Input tensor.

        Returns:
            :obj:`torch.Tensor`
        """
        for linear in self.linear_layers:
            input = linear(input)
            input = self.activation(input)
            input = self.dropout(input)
        return input


class LSTMBaseModule(nn.Module, FromConfig):
    """Base LSTM module."""

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor):
        raise NotImplementedError


class FullyConnectedLSTM(LSTMBaseModule):
    """
    FullyConnectedLSTM class.

    Args:
        input_size (:obj:`int`):
            The number of expected features in the input.
        hidden_size (:obj:`int`):
            The number of features in the hidden state.
        num_layers (:obj:`int`, optional, defaults to ``1``):
            Number of recurrent layers.
        dropout (:obj:`float`, optional, defaults to ``0.1``):
            Dropout hyper-parameter between layers.
        bidirectional (:obj:`bool`, optional, defaults to :obj:`True`):
            If ``True``, becomes a bidirectional LSTM.

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        # params
        self.output_size = input_size + num_layers * (2 * hidden_size)

        lstm_layers = []
        norm_layers = []
        dropout_layers = []
        cumulative_input_size = input_size

        self.input_normalization = nn.LayerNorm(input_size)

        for _ in range(num_layers):
            lstm_layers.append(
                nn.LSTM(
                    cumulative_input_size,
                    hidden_size,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
            )
            norm_layers.append(nn.LayerNorm(2 * hidden_size))
            dropout_layers.append(nn.Dropout(dropout))
            # update input size
            cumulative_input_size += 2 * hidden_size

        self.lstm_layers = nn.ModuleList(lstm_layers)
        self.norm_layers = nn.ModuleList(norm_layers)
        self.dropout_layers = nn.ModuleList(dropout_layers)

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor):
        """
        Forward implementation.

        Args:
            input (:obj:`torch.Tensor`):
                Input tensor.
            input_lengths (:obj:`torch.Tensor`):
                Tensor with the lengths for each row in ``input`` before padding.

        Returns:
            :obj:`torch.Tensor`

        """
        total_length = input.shape[1]
        x = self.input_normalization(input)

        for lstm, drop, norm in zip(self.lstm_layers, self.dropout_layers, self.norm_layers):
            packed_input = pack_padded_sequence(
                x, input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_encodings, _ = lstm(packed_input)
            encodings, _ = pad_packed_sequence(
                packed_encodings, total_length=total_length, batch_first=True
            )
            encodings = drop(encodings)
            encodings = norm(encodings)
            x = torch.cat([x, encodings], dim=-1)

        return x


class RoleEncoder(nn.Module, FromConfig):
    """
    Role encoder for the SRL model.

    Args:
        predicate_state_encoder:
        argument_state_encoder:
        argument_sequence_encoder:
        role_encoder:

    """

    def __init__(
        self,
        predicate_state_encoder: StateEncoder,
        argument_state_encoder: StateEncoder,
        argument_sequence_encoder: LSTMBaseModule,
        role_encoder: StateEncoder,
    ):
        super(RoleEncoder, self).__init__()
        # layers
        self.predicate_state_encoder = predicate_state_encoder
        self.argument_state_encoder = argument_state_encoder
        self.argument_sequence_encoder = argument_sequence_encoder
        self.role_encoder = role_encoder
        # params
        self.output_size = self.role_encoder.output_size

    def forward(
        self, inputs: torch.Tensor, sentence_lengths: torch.Tensor, predicate_indices: torch.Tensor
    ):
        """

        Args:
            inputs:
            sentence_lengths:
            predicate_indices:

        Returns:

        """
        # get batch sequences length
        timesteps = inputs.shape[1]
        # predicate states encodings
        predicate_state_encodings = self.predicate_state_encoder(inputs)
        predicate_state_encodings = predicate_state_encodings.unsqueeze(2)
        predicate_state_encodings = predicate_state_encodings.expand(-1, -1, timesteps, -1)
        # argument states encodings
        argument_state_encodings = self.argument_state_encoder(inputs)
        argument_state_encodings = argument_state_encodings.unsqueeze(1)
        argument_state_encodings = argument_state_encodings.expand(-1, timesteps, -1, -1)
        # predicate/argument aware encodings
        predicate_argument_states = torch.cat(
            (predicate_state_encodings, argument_state_encodings), dim=-1
        )
        # get predicate vectors
        predicate_argument_states = predicate_argument_states[predicate_indices == 1]
        # get row lengths and repeat by the number of predicates in the row
        predicates_counter = torch.sum(predicate_indices, dim=1)
        argument_sequence_lengths = torch.repeat_interleave(sentence_lengths, predicates_counter)
        max_argument_sequence_length = torch.max(argument_sequence_lengths)
        predicate_argument_states = predicate_argument_states[:, :max_argument_sequence_length, :]
        # arguments encoding
        argument_encodings = self.argument_sequence_encoder(
            predicate_argument_states, argument_sequence_lengths
        )
        return self.role_encoder(argument_encodings)
