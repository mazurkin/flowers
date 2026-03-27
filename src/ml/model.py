import dataclasses
import logging
import typing as t

import torch
import torch.nn

logger: logging.Logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class FlowersModelConfig:
    """Common configuration for the GAN encoder (discriminator) and decoder (generator).

    The encoder and decoder share image dimensions, channel count, and base filter count
    so that their architectures are symmetric. The encoder downsamples from image_size to 1x1
    through NUM_BLOCKS strided convolutions. The decoder upsamples from 1x1 to image_size
    through NUM_BLOCKS transposed convolutions.

    With image_size=64 and NUM_BLOCKS=4, spatial progression is:
        decoder: 1 -> 4 -> 8 -> 16 -> 32 -> 64
        encoder: 64 -> 32 -> 16 -> 8 -> 4 -> 1
    """

    # target square image size in pixels (must be divisible by 2^num_blocks)
    image_size: int = dataclasses.field(
        default=224,
        metadata={'help': 'Target square image size in pixels'},
    )

    # number of input/output image channels (RGB)
    image_channels: int = dataclasses.field(
        default=3,
        metadata={'help': 'Number of image channels (3 for RGB)'},
    )

    # dimensionality of the latent noise vector z
    latent_dim: int = dataclasses.field(
        default=128,
        metadata={'help': 'Dimensionality of the latent noise vector z'},
    )

    # base number of convolution filters; multiplied by powers of 2 in deeper layers
    base_filters: int = dataclasses.field(
        default=64,
        metadata={'help': 'Base number of convolution filters'},
    )

    # number of downsampling/upsampling blocks (each halves or doubles spatial dims)
    num_blocks: int = dataclasses.field(
        default=5,
        metadata={'help': 'Number of downsampling (encoder) / upsampling (decoder) blocks'},
    )

    # convolution kernel size used throughout the encoder and decoder
    kernel_size: int = dataclasses.field(
        default=4,
        metadata={'help': 'Convolution kernel size'},
    )

    # convolution stride used in strided/transposed convolutions
    stride: int = dataclasses.field(
        default=2,
        metadata={'help': 'Convolution stride for spatial down/upsampling'},
    )

    # convolution padding
    padding: int = dataclasses.field(
        default=1,
        metadata={'help': 'Convolution padding'},
    )

    # negative slope for LeakyReLU in the encoder
    encoder_leaky_relu_slope: float = dataclasses.field(
        default=0.2,
        metadata={'help': 'Negative slope for LeakyReLU in the encoder'},
    )

    def encoder_channels(self) -> list[int]:
        """Returns the per-block output channel counts for the encoder.

        The encoder doubles the channel count at each block starting from base_filters.

        Returns:
            list of output channel sizes, length == num_blocks
        """
        return [self.base_filters * (2 ** i) for i in range(self.num_blocks)]

    def decoder_channels(self) -> list[int]:
        """Returns the per-block output channel counts for the decoder.

        The decoder mirrors the encoder in reverse, halving channels at each block.

        Returns:
            list of output channel sizes, length == num_blocks
        """
        return [self.base_filters * (2 ** i) for i in range(self.num_blocks - 1, -1, -1)]

    def initial_spatial_size(self) -> int:
        """Returns the spatial size of the feature map at the deepest (bottleneck) layer.

        This is the image_size after num_blocks halvings: image_size / 2^num_blocks.

        Returns:
            spatial size at the bottleneck
        """
        size: int = self.image_size // (2 ** self.num_blocks)
        assert size >= 1, f'image_size={self.image_size} is too small for num_blocks={self.num_blocks}'

        return size


class FlowersModelEncoder(torch.nn.Module):
    """GAN discriminator (encoder) that maps an image to a real/fake scalar.

    Architecture:
        input Conv2d (no normalization) -> (num_blocks - 1) x [Conv2d -> BatchNorm -> LeakyReLU]
        -> final Conv2d -> Flatten -> scalar output

    With image_size=64, base_filters=64, num_blocks=4, the channel progression is:
        3 -> 64 -> 128 -> 256 -> 512 -> 1

    Spatial progression (stride=2):
        64 -> 32 -> 16 -> 8 -> 4 -> 1

    Parameters:
        config: FlowersModelConfig with shared architecture hyperparameters
    """

    def __init__(self, config: FlowersModelConfig) -> None:
        super().__init__()

        self.config: t.Final[FlowersModelConfig] = config

        channels: list[int] = config.encoder_channels()

        # build network
        self.net: t.Final[torch.nn.Sequential] = torch.nn.Sequential()

        # first layer: image_channels -> base_filters, no batch norm (DCGAN convention)
        self.net.append(
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=config.image_channels,
                    out_channels=channels[0],
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=config.padding,
                    bias=False,
                ),
                torch.nn.LeakyReLU(negative_slope=config.encoder_leaky_relu_slope, inplace=True)
            )
        )

        # intermediate blocks: each doubles channels, halves spatial dims
        for i in range(1, len(channels)):
            self.net.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=channels[i - 1],
                        out_channels=channels[i],
                        kernel_size=config.kernel_size,
                        stride=config.stride,
                        padding=config.padding,
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(channels[i]),
                    torch.nn.LeakyReLU(negative_slope=config.encoder_leaky_relu_slope, inplace=True)
                )
            )

        # final layer: collapse to a single scalar per sample
        # spatial size at this point is initial_spatial_size (e.g. 4x4)
        self.net.append(
            torch.nn.Conv2d(
                in_channels=channels[-1],
                out_channels=1,
                kernel_size=config.initial_spatial_size(),
                stride=1,
                padding=0,
                bias=False,
            )
        )

        total_params: int = sum(p.numel() for p in self.parameters())
        logger.info('FlowersModelEncoder total parameters: %s', total_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder (discriminator).

        Parameters:
            x: input images of shape (B, image_channels, image_size, image_size)

        Returns:
            raw logits of shape (B, 1) — not passed through sigmoid
        """
        batch_size: int = x.shape[0]
        assert x.shape == torch.Size([
            batch_size, self.config.image_channels, self.config.image_size, self.config.image_size,
        ])

        out: torch.Tensor = self.net(x)
        assert out.shape == torch.Size([batch_size, 1, 1, 1])

        # flatten from (B, 1, 1, 1) to (B, 1)
        logits: torch.Tensor = out.view(batch_size, 1)
        assert logits.shape == torch.Size([batch_size, 1])

        return logits


class FlowersModelDecoder(torch.nn.Module):
    """GAN generator (decoder) that maps a latent noise vector to an image.

    Architecture:
        Linear projection -> Reshape to (deepest_channels, initial_spatial, initial_spatial)
        -> (num_blocks - 1) x [ConvTranspose2d -> BatchNorm -> ReLU]
        -> final ConvTranspose2d -> Tanh

    With image_size=64, base_filters=64, latent_dim=128, num_blocks=4, the channel progression is:
        latent_dim -> 512 -> 256 -> 128 -> 64 -> 3

    Spatial progression (stride=2):
        1 -> 4 -> 8 -> 16 -> 32 -> 64

    Parameters:
        config: FlowersModelConfig with shared architecture hyperparameters
    """

    def __init__(self, config: FlowersModelConfig) -> None:
        super().__init__()

        self.config: t.Final[FlowersModelConfig] = config

        channels: list[int] = config.decoder_channels()

        # the deepest (most compressed) channel count, matching the encoder bottleneck
        self.deepest_channels: t.Final[int] = channels[0]

        # spatial size after projection reshape
        self.initial_spatial: t.Final[int] = config.initial_spatial_size()

        # project latent vector to the spatial feature map
        self.project: t.Final[torch.nn.Sequential] = \
            torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=config.latent_dim,
                    out_features=self.deepest_channels * self.initial_spatial * self.initial_spatial,
                    bias=False,
                ),
                torch.nn.BatchNorm1d(self.deepest_channels * self.initial_spatial * self.initial_spatial),
                torch.nn.ReLU(inplace=True),
            )

        # build network
        self.net: t.Final[torch.nn.Sequential] = torch.nn.Sequential()

        # intermediate upsampling blocks: each halves channels, doubles spatial dims
        for i in range(1, len(channels)):
            self.net.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(
                        in_channels=channels[i - 1],
                        out_channels=channels[i],
                        kernel_size=config.kernel_size,
                        stride=config.stride,
                        padding=config.padding,
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(channels[i]),
                    torch.nn.ReLU(inplace=True),
                )
            )

        # final layer: upsample to image_channels with Tanh activation for [-1, 1] output
        self.net.append(
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    in_channels=channels[-1],
                    out_channels=config.image_channels,
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=config.padding,
                    bias=False,
                ),
                torch.nn.Tanh()
            )
        )

        total_params: int = sum(p.numel() for p in self.parameters())
        logger.info('FlowersModelDecoder total parameters: %s', total_params)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder (generator).

        Parameters:
            z: latent noise vector of shape (B, latent_dim)

        Returns:
            generated images of shape (B, image_channels, image_size, image_size) in [-1, 1]
        """
        batch_size: int = z.shape[0]
        assert z.shape == torch.Size([batch_size, self.config.latent_dim])

        # project and reshape to spatial feature map
        projected: torch.Tensor = self.project(z)
        assert projected.shape == torch.Size([
            batch_size, self.deepest_channels * self.initial_spatial * self.initial_spatial,
        ])

        reshaped: torch.Tensor = projected.view(
            batch_size, self.deepest_channels, self.initial_spatial, self.initial_spatial,
        )
        assert reshaped.shape == torch.Size([
            batch_size, self.deepest_channels, self.initial_spatial, self.initial_spatial,
        ])

        # upsample through transposed convolutions
        images: torch.Tensor = self.net(reshaped)
        assert images.shape == torch.Size([
            batch_size, self.config.image_channels, self.config.image_size, self.config.image_size,
        ])

        return images
