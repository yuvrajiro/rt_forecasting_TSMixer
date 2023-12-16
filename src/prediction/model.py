from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch import nn
from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not
from darts.models.forecasting.pl_forecasting_module import (
    PLMixedCovariatesModule,
    io_processor,
)
from darts.models.forecasting.tft_submodels import get_embedding_size
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.data import (
    MixedCovariatesSequentialDataset,
    MixedCovariatesTrainingDataset,
    TrainingDataset,
)
logger = get_logger(__name__)
MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class TimeBatchNorm2d(nn.BatchNorm1d):
    """A batch normalization layer that normalizes over the last two dimensions of a
    sequence in PyTorch, mimicking Keras behavior.

    This class extends nn.BatchNorm1d to apply batch normalization across time and
    feature dimensions.

    Attributes:
        num_time_steps (int): Number of time steps in the input.
        num_channels (int): Number of channels in the input.
    """

    def __init__(self, normalized_shape: tuple[int, int]):
        """Initializes the TimeBatchNorm2d module.

        Args:
            normalized_shape (tuple[int, int]): A tuple (num_time_steps, num_channels)
                representing the shape of the time and feature dimensions to normalize.
        """
        num_time_steps, num_channels = normalized_shape
        super().__init__(num_channels * num_time_steps)
        self.num_time_steps = num_time_steps
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:
        """Applies the batch normalization over the last two dimensions of the input tensor.

        Args:
            x (Tensor): A 3D tensor with shape (N, S, C), where N is the batch size,
                S is the number of time steps, and C is the number of channels.

        Returns:
            Tensor: A 3D tensor with batch normalization applied over the last two dims.

        Raises:
            ValueError: If the input tensor is not 3D.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input tensor, but got {x.ndim}D tensor instead.")

        # Reshaping input to combine time and feature dimensions for normalization
        x = x.reshape(x.shape[0], -1, 1)

        # Applying batch normalization
        x = super().forward(x)

        # Reshaping back to original dimensions (N, S, C)
        x = x.reshape(x.shape[0], self.num_time_steps, self.num_channels)

        return x


class FeatureMixing(nn.Module):
    """A module for feature mixing with flexibility in normalization and activation.

    This module provides options for batch normalization before or after mixing features,
    uses dropout for regularization, and allows for different activation functions.

    Args:
        sequence_length: The length of the sequences to be transformed.
        input_channels: The number of input channels to the module.
        output_channels: The number of output channels from the module.
        ff_dim: The dimension of the feed-forward network internal to the module.
        activation_fn: The activation function used within the feed-forward network.
        dropout_rate: The dropout probability used for regularization.
        normalize_before: A boolean indicating whether to apply normalization before
            the rest of the operations.
    """

    def __init__(
            self,
            sequence_length: int,
            input_channels: int,
            output_channels: int,
            ff_dim: int,
            activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
            dropout_rate: float = 0.1,
            normalize_before: bool = True,
            norm_type: type[nn.Module] = TimeBatchNorm2d,
    ):
        """Initializes the FeatureMixing module with the provided parameters."""
        super().__init__()

        self.norm_before = (
            norm_type((sequence_length, input_channels))
            if normalize_before
            else nn.Identity()
        )
        self.norm_after = (
            norm_type((sequence_length, output_channels))
            if not normalize_before
            else nn.Identity()
        )

        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_channels, ff_dim)
        self.fc2 = nn.Linear(ff_dim, output_channels)

        self.projection = (
            nn.Linear(input_channels, output_channels)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FeatureMixing module.

        Args:
            x: A 3D tensor with shape (N, C, L) where C is the channel dimension.

        Returns:
            The output tensor after feature mixing.
        """

        x_proj = self.projection(x)

        x = self.norm_before(x)

        x = self.fc1(x)  # Apply the first linear transformation.
        x = self.activation_fn(x)  # Apply the activation function.
        x = self.dropout(x)  # Apply dropout for regularization.
        x = self.fc2(x)  # Apply the second linear transformation.
        x = self.dropout(x)  # Apply dropout again if needed.

        x = x_proj + x  # Add the projection shortcut to the transformed features.

        return self.norm_after(x)


class ConditionalFeatureMixing(nn.Module):
    """Conditional feature mixing module that incorporates static features.

    This module extends the feature mixing process by including static features. It uses
    a linear transformation to integrate static features into the dynamic feature space,
    then applies the feature mixing on the concatenated features.

    Args:
        input_channels: The number of input channels of the dynamic features.
        output_channels: The number of output channels after feature mixing.
        static_channels: The number of channels in the static feature input.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in feature mixing.
        dropout_rate: The dropout probability used in the feature mixing operation.
    """

    def __init__(
            self,
            sequence_length: int,
            input_channels: int,
            output_channels: int,
            static_channels: int,
            ff_dim: int,
            activation_fn: Callable = F.relu,
            dropout_rate: float = 0.1,
            normalize_before: bool = False,
            norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.fr_static = nn.Linear(static_channels, output_channels)
        self.fm = FeatureMixing(
            sequence_length,
            input_channels + output_channels,
            output_channels,
            ff_dim,
            activation_fn,
            dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    def forward(
            self, x: torch.Tensor, x_static: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies conditional feature mixing using both dynamic and static inputs.

        Args:
            x: A tensor representing dynamic features, typically with shape
               [batch_size, time_steps, input_channels].
            x_static: A tensor representing static features, typically with shape
               [batch_size, static_channels].

        Returns:
            A tuple containing:
            - The output tensor after applying conditional feature mixing.
            - The transformed static features tensor for monitoring or further processing.
        """
        v = self.fr_static(x_static)  # Transform static features to match output channels.

        v = v.unsqueeze(1).repeat(
            1, x.shape[1], 1
        )  # Repeat static features across time steps.

        return (
            self.fm(
                torch.cat([x, v], dim=-1)
            ),  # Apply feature mixing on concatenated features.
            v.detach(),  # Return detached static feature for monitoring or further use.
        )


class TimeMixing(nn.Module):
    """Applies a transformation over the time dimension of a sequence.

    This module applies a linear transformation followed by an activation function
    and dropout over the sequence length of the input feature tensor after converting
    feature maps to the time dimension and then back.

    Args:
        input_channels: The number of input channels to the module.
        sequence_length: The length of the sequences to be transformed.
        activation_fn: The activation function to be used after the linear transformation.
        dropout_rate: The dropout probability to be used after the activation function.
    """

    def __init__(
            self,
            sequence_length: int,
            input_channels: int,
            activation_fn: Callable = F.relu,
            dropout_rate: float = 0.1,
            norm_type: type[nn.Module] = TimeBatchNorm2d,
    ):
        """Initializes the TimeMixing module with the specified parameters."""
        super().__init__()
        self.norm = norm_type((sequence_length, input_channels))
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(sequence_length, sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the time mixing operations on the input tensor.

        Args:
            x: A 3D tensor with shape (N, C, L), where C = channel dimension and
                L = sequence length.

        Returns:
            The normalized output tensor after time mixing transformations.
        """
        x_temp = feature_to_time(x)  # Convert feature maps to time dimension. Assumes definition elsewhere.
        x_temp = self.activation_fn(self.fc1(x_temp))
        x_temp = self.dropout(x_temp)
        x_res = time_to_feature(x_temp)  # Convert back from time to feature maps.

        return self.norm(x + x_res)  # Apply normalization and combine with original input.


class MixerLayer(nn.Module):
    """A residual block that combines time and feature mixing for sequence data.

    This module sequentially applies time mixing and feature mixing, which are forms
    of data augmentation and feature transformation that can help in learning temporal
    dependencies and feature interactions respectively.

    Args:
        sequence_length: The length of the input sequences.
        input_channels: The number of input channels to the module.
        output_channels: The number of output channels from the module.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in both time and feature mixing.
        dropout_rate: The dropout probability used in both mixing operations.
    """

    def __init__(
            self,
            sequence_length: int,
            input_channels: int,
            output_channels: int,
            ff_dim: int,
            activation_fn: Callable = F.relu,
            dropout_rate: float = 0.1,
            normalize_before: bool = False,
            norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        """Initializes the MixLayer with time and feature mixing modules."""
        super().__init__()

        self.time_mixing = TimeMixing(
            sequence_length,
            input_channels,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
        )
        self.feature_mixing = FeatureMixing(
            sequence_length,
            input_channels,
            output_channels,
            ff_dim,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
            normalize_before=normalize_before,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MixLayer module.

        Args:
            x: A 3D tensor with shape (N, C, L) to be processed by the mixing layers.

        Returns:
            The output tensor after applying time and feature mixing operations.
        """
        x = self.time_mixing(x)  # Apply time mixing first.
        x = self.feature_mixing(x)  # Then apply feature mixing.

        return x


class ConditionalMixerLayer(nn.Module):
    """Conditional mix layer combining time and feature mixing with static context.

    This module combines time mixing and conditional feature mixing, where the latter
    is influenced by static features. This allows the module to learn representations
    that are influenced by both dynamic and static features.

    Args:
        sequence_length: The length of the input sequences.
        input_channels: The number of input channels of the dynamic features.
        output_channels: The number of output channels after feature mixing.
        static_channels: The number of channels in the static feature input.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in both mixing operations.
        dropout_rate: The dropout probability used in both mixing operations.
    """

    def __init__(
            self,
            sequence_length: int,
            input_channels: int,
            output_channels: int,
            static_channels: int,
            ff_dim: int,
            activation_fn: Callable = F.relu,
            dropout_rate: float = 0.1,
            normalize_before: bool = False,
            norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.time_mixing = TimeMixing(
            sequence_length,
            input_channels,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
        )
        self.feature_mixing = ConditionalFeatureMixing(
            sequence_length,
            input_channels,
            output_channels=output_channels,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    def forward(self, x: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """Forward pass for the conditional mix layer.

        Args:
            x: A tensor representing dynamic features, typically with shape
               [batch_size, time_steps, input_channels].
            x_static: A tensor representing static features, typically with shape
               [batch_size, static_channels].

        Returns:
            The output tensor after applying time and conditional feature mixing.
        """
        x = self.time_mixing(x)  # Apply time mixing first.
        x, _ = self.feature_mixing(x, x_static)  # Then apply conditional feature mixing.

        return x


def time_to_feature(x: torch.Tensor) -> torch.Tensor:
    """Converts a time series tensor to a feature tensor."""
    return x.permute(0, 2, 1)


feature_to_time = time_to_feature


class _TSMixer(PLMixedCovariatesModule):
    def __init__(
            self,
            n_input_channels: int,
            n_extra_channels: int,
            output_dim: Tuple[int, int],
            variables_meta: Dict[str, Dict[str, List[str]]],
            num_static_components: int,
            hidden_size: Union[int, List[int]],
            num_block: int,
            hidden_continuous_size: int,
            dropout: float,
            add_relative_index: bool,
            norm_type: Union[str, nn.Module],
            **kwargs,
    ):

        """
        A metaclass for all TSMixer models.

        Parameters
        ----------
        n_input_channels
            number of input channels
        n_extra_channels
            number of extra channels
        output_dim
            output dimension
        variables_meta
            variables meta
        num_static_components
            number of static components
        hidden_size
            hidden size
        num_block
            number of blocks
        hidden_continuous_size
            hidden continuous size
        dropout
            dropout rate
        add_relative_index
            whether to add relative index
        norm_type
            type of normalization
        kwargs
            all parameters required for :class:`darts.model.forecasting_models.TorchForecastingModel` base class.

        """

        super().__init__(**kwargs)

        self.n_targets, self.loss_size = output_dim
        self.variables_meta = variables_meta
        self.num_static_components = num_static_components
        self.hidden_size = hidden_size
        self.hidden_continuous_size = hidden_continuous_size
        self.num_block = num_block
        self.dropout = dropout
        self.add_relative_index = add_relative_index

        # initialize last batch size to check if new mask needs to be generated
        self.batch_size_last = -1
        self.relative_index = None

        # can be edited to use static variable
        static_channels = 1

        activation_fn = kwargs.get("activation_fn", "relu")
        if hasattr(F, activation_fn):
            activation_fn = getattr(F, activation_fn)
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")


        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm
        sequence_length = self.input_chunk_length
        prediction_length = self.output_chunk_length
        output_channels = self.n_targets
        input_channels = n_input_channels
        extra_channels = n_extra_channels
        dropout_rate = self.dropout
        normalize_before = False
        num_blocks = self.num_block
        ff_dim = self.hidden_size
        self.fc_hist = nn.Linear(sequence_length, prediction_length)
        self.fc_out = nn.Linear(self.hidden_size, output_channels)

        self.feature_mixing_hist = ConditionalFeatureMixing(
            sequence_length=prediction_length,
            input_channels=input_channels + extra_channels,
            output_channels=self.hidden_size,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )
        self.feature_mixing_future = ConditionalFeatureMixing(
            sequence_length=prediction_length,
            input_channels=extra_channels,
            output_channels=self.hidden_size,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

        self.conditional_mixer = self._build_mixer(
            num_blocks,
            self.hidden_size,
            prediction_length,
            ff_dim=ff_dim,
            static_channels=static_channels,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    @staticmethod
    def _build_mixer(
            num_blocks: int, hidden_channels: int, prediction_length: int, **kwargs
    ):
        """Build the mixer blocks for the model."""
        channels = [2 * hidden_channels] + [hidden_channels] * (num_blocks - 1)

        return nn.ModuleList(
            [
                ConditionalMixerLayer(
                    input_channels=in_ch,
                    output_channels=out_ch,
                    sequence_length=prediction_length,
                    **kwargs,
                )
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    @property
    def reals(self) -> List[str]:
        """
        List of all continuous variables in model
        """
        return self.variables_meta["model_config"]["reals_input"]

    @property
    def static_variables(self) -> List[str]:
        """
        List of all static variables in model
        """
        return self.variables_meta["model_config"]["static_input"]

    @property
    def numeric_static_variables(self) -> List[str]:
        """
        List of numeric static variables in model
        """
        return self.variables_meta["model_config"]["static_input_numeric"]

    @property
    def categorical_static_variables(self) -> List[str]:
        """
        List of categorical static variables in model
        """
        return self.variables_meta["model_config"]["static_input_categorical"]

    @property
    def encoder_variables(self) -> List[str]:
        """
        List of all encoder variables in model (excluding static variables)
        """
        return self.variables_meta["model_config"]["time_varying_encoder_input"]

    @property
    def decoder_variables(self) -> List[str]:
        """
        List of all decoder variables in model (excluding static variables)
        """
        return self.variables_meta["model_config"]["time_varying_decoder_input"]

    @staticmethod
    def expand_static_context(context: torch.Tensor, time_steps: int) -> torch.Tensor:
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, time_steps, -1)

    @staticmethod
    def get_relative_index(
            encoder_length: int,
            decoder_length: int,
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
    ) -> torch.Tensor:
        """
        Returns scaled time index relative to prediction point.
        """
        index = torch.arange(
            encoder_length + decoder_length, dtype=dtype, device=device
        )
        prediction_index = encoder_length - 1
        index[:encoder_length] = index[:encoder_length] / prediction_index
        index[encoder_length:] = index[encoder_length:] / prediction_index
        return index.reshape(1, len(index), 1).repeat(batch_size, 1, 1)

    @staticmethod
    def get_attention_mask_full(
            time_steps: int, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """
        Returns causal mask to apply for self-attention layer.
        """
        eye = torch.eye(time_steps, dtype=dtype, device=device)
        mask = torch.cumsum(eye.unsqueeze(0).repeat(batch_size, 1, 1), dim=1)
        return mask < 1

    @staticmethod
    def get_attention_mask_future(
            encoder_length: int,
            decoder_length: int,
            batch_size: int,
            device: str,
            full_attention: bool,
    ) -> torch.Tensor:
        """
        Returns causal mask to apply for self-attention layer that acts on future input only.
        The model will attend to all `False` values.
        """
        if full_attention:
            # attend to entire past and future input
            decoder_mask = torch.zeros(
                (decoder_length, decoder_length), dtype=torch.bool, device=device
            )
        else:
            # attend only to past steps relative to forecasting step in the future
            # indices to which is attended
            attend_step = torch.arange(decoder_length, device=device)
            # indices for which is predicted
            predict_step = torch.arange(0, decoder_length, device=device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = attend_step >= predict_step
        # attend to all past input
        encoder_mask = torch.zeros(
            batch_size, encoder_length, dtype=torch.bool, device=device
        )
        # combine masks along attended time - first encoder and then decoder

        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(batch_size, -1, -1),
            ),
            dim=2,
        )
        return mask

    @io_processor
    def forward(
            self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """TFT model forward pass.

        Parameters
        ----------
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`
            is the output/future chunk. Input dimensions are `(n_samples, n_time_steps, n_variables)`

        Returns
        -------
        torch.Tensor
            the output tensor
        """
        x_cont_past, x_cont_future, x_static = x_in
        dim_samples, dim_time, dim_variable = 0, 1, 2
        device = x_in[0].device

        batch_size = x_cont_past.shape[dim_samples]
        encoder_length = self.input_chunk_length
        decoder_length = self.output_chunk_length
        time_steps = encoder_length + decoder_length

        # avoid unnecessary regeneration of attention mask
        if batch_size != self.batch_size_last:
            if self.add_relative_index:
                self.relative_index = self.get_relative_index(
                    encoder_length=encoder_length,
                    decoder_length=decoder_length,
                    batch_size=batch_size,
                    device=device,
                    dtype=x_cont_past.dtype,
                )

            self.batch_size_last = batch_size

        if self.add_relative_index:
            x_cont_past = torch.cat(
                [
                    ts[:, :encoder_length, :]
                    for ts in [x_cont_past, self.relative_index]
                    if ts is not None
                ],
                dim=dim_variable,
            )
            x_cont_future = torch.cat(
                [
                    ts[:, -decoder_length:, :]
                    for ts in [x_cont_future, self.relative_index]
                    if ts is not None
                ],
                dim=dim_variable,
            )

        input_vectors_past = {
            name: x_cont_past[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.encoder_variables)
        }
        input_vectors_future = {
            name: x_cont_future[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.decoder_variables)
        }

        # Embedding and variable selection
        if self.static_variables:
            # categorical static covariate embeddings
            if self.categorical_static_variables:
                static_embedding = self.input_embeddings(
                    torch.cat(
                        [
                            x_static[:, :, idx]
                            for idx, name in enumerate(self.static_variables)
                            if name in self.categorical_static_variables
                        ],
                        dim=1,
                    ).int()
                )
            else:
                static_embedding = {}
            # add numerical static covariates
            static_embedding.update(
                {
                    name: x_static[:, :, idx]
                    for idx, name in enumerate(self.static_variables)
                    if name in self.numeric_static_variables
                }
            )
            static_embedding, static_covariate_var = self.static_covariates_vsn(
                static_embedding
            )
        else:
            static_embedding = torch.zeros(
                (x_cont_past.shape[0], self.hidden_size),
                dtype=x_cont_past.dtype,
                device=device,
            )
            static_covariate_var = None

        # Concatenate historical time series data with additional historical data
        x_hist = x_cont_past
        x_static = torch.zeros([x_hist.size(0), 1], dtype=torch.float64)
        x_hist_temp = feature_to_time(x_hist)
        x_hist_temp = self.fc_hist(x_hist_temp)
        x_hist = time_to_feature(x_hist_temp)
        x_hist, _ = self.feature_mixing_hist(x_hist, x_static=x_static)
        x_future, _ = self.feature_mixing_future(x_cont_future, x_static=x_static)
        x = torch.cat([x_hist, x_future], dim=-1)
        for mixing_layer in self.conditional_mixer:
            x = mixing_layer(x, x_static=x_static)
        x = self.fc_out(x).unsqueeze(-1)
        return x


class TSMixer(MixedCovariatesTorchModel):
    def __init__(
            self,
            input_chunk_length: int,
            output_chunk_length: int,
            n_input_channels: int,
            n_extra_channels: int,
            hidden_size: Union[int, List[int]] = 16,
            num_block: int = 10,
            dropout: float = 0.1,
            hidden_continuous_size: int = 8,
            add_relative_index: bool = False,
            norm_type: Union[str, nn.Module] = "batch",
            use_static_covariates: bool = True,
            **kwargs,
    ):
        """
        PyTorch implementation of the New TSMixer model from `this paper <https://arxiv.org/pdf/2303.06053.pdf>`_.
        The implemntation use Darts Forescasting Module, this class works as a plugin for Darts.

        Parameters
        ----------
        input_chunk_length
            length of the input chunk
        output_chunk_length
            length of the output chunk
        n_input_channels
            number of input channels
        n_extra_channels
            number of extra channels
        hidden_size
            hidden size of the model
        num_block
            number of blocks
        dropout
            dropout rate
        hidden_continuous_size
            hidden size for continuous variables
        add_relative_index
            whether to add relative index
        norm_type
            type of normalization
        use_static_covariates
            whether to use static covariates
        kwargs
            all parameters required for :class:`darts.model.forecasting_models.TorchForecastingModel` base class.



        """

        model_kwargs = {key: val for key, val in self.model_params.items()}

        model_kwargs["loss_fn"] = nn.MSELoss()
        super().__init__(**self._extract_torch_model_params(**model_kwargs))
        self.pl_module_params = self._extract_pl_module_params(**model_kwargs)
        self.hidden_size = hidden_size
        self.num_block = num_block
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.n_input_channels = n_input_channels
        self.n_extra_channels = n_extra_channels
        self.add_relative_index = add_relative_index
        self.output_dim: Optional[Tuple[int, int]] = None
        self.norm_type = norm_type
        self._considers_static_covariates = use_static_covariates

    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        """
        `train_sample` contains the following tensors:
            (past_target, past_covariates, historic_future_covariates, future_covariates, static_covariates,
            future_target)

            each tensor has shape (n_timesteps, n_variables)
            - past/historic tensors have shape (input_chunk_length, n_variables)
            - future tensors have shape (output_chunk_length, n_variables)
            - static covariates have shape (component, static variable)

        Darts Interpretation of pytorch-forecasting's TimeSeriesDataSet:
            time_varying_knowns : future_covariates (including historic_future_covariates)
            time_varying_unknowns : past_targets, past_covariates

            time_varying_encoders : [past_targets, past_covariates, historic_future_covariates, future_covariates]
            time_varying_decoders : [historic_future_covariates, future_covariates]

        `variable_meta` is used in TFT to access specific variables
        """
        (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariates,
            future_target,
        ) = train_sample

        # add a covariate placeholder so that relative index will be included
        if self.add_relative_index:
            time_steps = self.input_chunk_length + self.output_chunk_length

            expand_future_covariate = np.arange(time_steps).reshape((time_steps, 1))

            historic_future_covariate = np.concatenate(
                [
                    ts[: self.input_chunk_length]
                    for ts in [historic_future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
            future_covariate = np.concatenate(
                [
                    ts[-self.output_chunk_length:]
                    for ts in [future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )

        self.output_dim = (
            (future_target.shape[1], 1)
            if self.likelihood is None
            else (future_target.shape[1], self.likelihood.num_parameters)
        )

        tensors = [
            past_target,
            past_covariate,
            historic_future_covariate,  # for time varying encoders
            future_covariate,
            future_target,  # for time varying decoders
            static_covariates,  # for static encoder
        ]
        type_names = [
            "past_target",
            "past_covariate",
            "historic_future_covariate",
            "future_covariate",
            "future_target",
            "static_covariate",
        ]
        variable_names = [
            "target",
            "past_covariate",
            "future_covariate",
            "future_covariate",
            "target",
            "static_covariate",
        ]

        variables_meta = {
            "input": {
                type_name: [f"{var_name}_{i}" for i in range(tensor.shape[1])]
                for type_name, var_name, tensor in zip(
                    type_names, variable_names, tensors
                )
                if tensor is not None
            },
            "model_config": {},
        }

        reals_input = []
        categorical_input = []
        time_varying_encoder_input = []
        time_varying_decoder_input = []
        static_input = []
        static_input_numeric = []
        static_input_categorical = []
        categorical_embedding_sizes = {}
        for input_var in type_names:
            if input_var in variables_meta["input"]:
                vars_meta = variables_meta["input"][input_var]
                if input_var in [
                    "past_target",
                    "past_covariate",
                    "historic_future_covariate",
                ]:
                    time_varying_encoder_input += vars_meta
                    reals_input += vars_meta
                elif input_var in ["future_covariate"]:
                    time_varying_decoder_input += vars_meta
                    reals_input += vars_meta
                elif input_var in ["static_covariate"]:
                    if (
                            self.static_covariates is None
                    ):  # when training with fit_from_dataset
                        static_cols = pd.Index(
                            [i for i in range(static_covariates.shape[1])]
                        )
                    else:
                        static_cols = self.static_covariates.columns
                    numeric_mask = ~static_cols.isin(self.categorical_embedding_sizes)
                    for idx, (static_var, col_name, is_numeric) in enumerate(
                            zip(vars_meta, static_cols, numeric_mask)
                    ):
                        static_input.append(static_var)
                        if is_numeric:
                            static_input_numeric.append(static_var)
                            reals_input.append(static_var)
                        else:
                            # get embedding sizes for each categorical variable
                            embedding = self.categorical_embedding_sizes[col_name]
                            raise_if_not(
                                isinstance(embedding, (int, tuple)),
                                "Dict values of `categorical_embedding_sizes` must either be integers or tuples. Read "
                                "the TSMixer documentation for more information.",
                                logger,
                            )
                            if isinstance(embedding, int):
                                embedding = (embedding, get_embedding_size(n=embedding))
                            categorical_embedding_sizes[vars_meta[idx]] = embedding

                            static_input_categorical.append(static_var)
                            categorical_input.append(static_var)

        variables_meta["model_config"]["reals_input"] = list(dict.fromkeys(reals_input))
        variables_meta["model_config"]["categorical_input"] = list(
            dict.fromkeys(categorical_input)
        )
        variables_meta["model_config"]["time_varying_encoder_input"] = list(
            dict.fromkeys(time_varying_encoder_input)
        )
        variables_meta["model_config"]["time_varying_decoder_input"] = list(
            dict.fromkeys(time_varying_decoder_input)
        )
        variables_meta["model_config"]["static_input"] = list(
            dict.fromkeys(static_input)
        )
        variables_meta["model_config"]["static_input_numeric"] = list(
            dict.fromkeys(static_input_numeric)
        )
        variables_meta["model_config"]["static_input_categorical"] = list(
            dict.fromkeys(static_input_categorical)
        )

        n_static_components = (
            len(static_covariates) if static_covariates is not None else 0
        )

        self.categorical_embedding_sizes = categorical_embedding_sizes

        return _TSMixer(
            n_input_channels=self.n_input_channels,
            n_extra_channels=self.n_extra_channels,
            output_dim=self.output_dim,
            variables_meta=variables_meta,
            num_static_components=n_static_components,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            num_block=self.num_block,
            hidden_continuous_size=self.hidden_continuous_size,
            add_relative_index=self.add_relative_index,
            norm_type=self.norm_type,
            **self.pl_module_params,
        )

    def _build_train_dataset(
            self,
            target: Sequence[TimeSeries],
            past_covariates: Optional[Sequence[TimeSeries]],
            future_covariates: Optional[Sequence[TimeSeries]],
            max_samples_per_ts: Optional[int],
    ) -> MixedCovariatesSequentialDataset:

        raise_if(
            future_covariates is None and not self.add_relative_index,
            "TSMixer requires future covariates. The model applies multi-head attention queries on future "
            "inputs. Consider specifying a future encoder with `add_encoders` or setting `add_relative_index` "
            "to `True` at model creation (read TFT model docs for more information). "
            "These will automatically generate `future_covariates` from indexes.",
            logger,
        )

        return MixedCovariatesSequentialDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=self.uses_static_covariates,
        )

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        raise_if_not(
            isinstance(train_dataset, MixedCovariatesTrainingDataset),
            "TSMixer requires a training dataset of type MixedCovariatesTrainingDataset.",
        )

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def supports_static_covariates(self) -> bool:
        return True


