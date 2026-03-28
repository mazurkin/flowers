import typing as t

import albumentations
import numpy
import pytest
import torch
import torch.utils.data

import PIL.Image

import datasets

from ml.data import GanDataModuleConfig, GanDataset, GanDataModule
from ml.model import GanModelConfig, GanModelEncoder, GanModelDecoder
from ml.module import GanModule

# default image size used in tests (must be divisible by 2^num_blocks)
TEST_IMAGE_SIZE: t.Final[int] = 64

# number of RGB channels
NUM_CHANNELS: t.Final[int] = 3

# number of synthetic samples in the fake dataset
NUM_SAMPLES: t.Final[int] = 100


def make_test_config() -> GanDataModuleConfig:
    """Creates a small GanDataModuleConfig for testing."""
    return GanDataModuleConfig(
        image_height=TEST_IMAGE_SIZE,
        image_width=TEST_IMAGE_SIZE,
        batch_size=4,
        num_workers=0,
    )


def make_fake_hf_dataset(num_samples: int = NUM_SAMPLES) -> datasets.Dataset:
    """Creates a fake HuggingFace dataset mimicking huggan/metfaces.

    Parameters:
        num_samples: number of synthetic samples to generate

    Returns:
        a HuggingFace Dataset with 'image' (PIL) column
    """
    images: list[PIL.Image.Image] = []

    for i in range(num_samples):
        # create synthetic RGB images with random pixel values
        pixels: numpy.ndarray = numpy.random.randint(0, 256, (32, 32, NUM_CHANNELS), dtype=numpy.uint8)
        img: PIL.Image.Image = PIL.Image.fromarray(pixels)
        images.append(img)

    return datasets.Dataset.from_dict({
        'image': images,
    })


class TestGanDataConfig:

    def test_default_config(self) -> None:
        """Verifies default config values."""
        config: GanDataModuleConfig = GanDataModuleConfig()
        assert config.image_height == 256
        assert config.image_width == 256
        assert config.batch_size == 64
        assert config.eval_fraction == 0.05

    def test_frozen(self) -> None:
        """Verifies the config dataclass is immutable."""
        config: GanDataModuleConfig = GanDataModuleConfig()
        with pytest.raises(AttributeError):
            config.batch_size = 64


class TestGanDataset:

    def test_getitem_returns_correct_shape(self) -> None:
        """Verifies that a single sample has correct tensor shapes."""
        config: GanDataModuleConfig = make_test_config()
        hf_dataset: datasets.Dataset = make_fake_hf_dataset(num_samples=10)

        dm: GanDataModule = GanDataModule(config)
        transform: albumentations.Compose = dm.build_eval_transform()

        ds: GanDataset = GanDataset(hf_dataset, transform)

        assert len(ds) == 10

        sample: dict[str, torch.Tensor] = ds[0]

        pixel_values: torch.Tensor = sample['pixel_values']
        assert pixel_values.shape == torch.Size([NUM_CHANNELS, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE])
        assert pixel_values.dtype == torch.float32

    def test_grayscale_image_converted_to_rgb(self) -> None:
        """Verifies that a grayscale image is properly converted to 3-channel RGB."""
        config: GanDataModuleConfig = make_test_config()

        # create a single-channel grayscale image
        gray_pixels: numpy.ndarray = numpy.random.randint(0, 256, (32, 32), dtype=numpy.uint8)
        gray_image: PIL.Image.Image = PIL.Image.fromarray(gray_pixels, mode='L')
        hf_dataset: datasets.Dataset = datasets.Dataset.from_dict({
            'image': [gray_image],
        })

        dm: GanDataModule = GanDataModule(config)
        transform: albumentations.Compose = dm.build_eval_transform()
        ds: GanDataset = GanDataset(hf_dataset, transform)

        sample: dict[str, torch.Tensor] = ds[0]
        assert sample['pixel_values'].shape == torch.Size([NUM_CHANNELS, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE])

    def test_train_transform_produces_correct_shape(self) -> None:
        """Verifies the training augmentation pipeline produces correct output shape."""
        config: GanDataModuleConfig = make_test_config()
        hf_dataset: datasets.Dataset = make_fake_hf_dataset(num_samples=5)

        dm: GanDataModule = GanDataModule(config)
        transform: albumentations.Compose = dm.build_train_transform()
        ds: GanDataset = GanDataset(hf_dataset, transform)

        for i in range(len(ds)):
            sample: dict[str, torch.Tensor] = ds[i]
            assert sample['pixel_values'].shape == torch.Size([NUM_CHANNELS, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE])


class TestGanDataModule:

    def test_setup_creates_datasets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies that setup() creates train and eval datasets with correct splits."""
        fake_dataset: datasets.Dataset = make_fake_hf_dataset(num_samples=NUM_SAMPLES)

        # mock the HuggingFace load_dataset call to use our fake dataset
        monkeypatch.setattr(
            datasets,
            'load_dataset',
            lambda *args, **kwargs: fake_dataset,
        )

        config: GanDataModuleConfig = make_test_config()
        dm: GanDataModule = GanDataModule(config)
        dm.setup(stage=None)

        assert dm.train_dataset is not None
        assert dm.eval_dataset is not None

        total: int = len(dm.train_dataset) + len(dm.eval_dataset)
        assert total == NUM_SAMPLES

        # verify approximate split ratios (allow +-2 for rounding)
        expected_eval: int = int(NUM_SAMPLES * config.eval_fraction)
        assert abs(len(dm.eval_dataset) - expected_eval) <= 2

    def test_dataloaders_produce_batches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies that dataloaders produce correctly shaped batches."""
        fake_dataset: datasets.Dataset = make_fake_hf_dataset(num_samples=NUM_SAMPLES)

        monkeypatch.setattr(
            datasets,
            'load_dataset',
            lambda *args, **kwargs: fake_dataset,
        )

        config: GanDataModuleConfig = make_test_config()
        dm: GanDataModule = GanDataModule(config)
        dm.setup(stage=None)

        # check train dataloader
        train_dl: torch.utils.data.DataLoader = dm.train_dataloader()
        batch: dict[str, torch.Tensor] = next(iter(train_dl))

        assert batch['pixel_values'].shape == torch.Size([
            config.batch_size,
            NUM_CHANNELS,
            TEST_IMAGE_SIZE,
            TEST_IMAGE_SIZE,
        ])

        # check val dataloader
        val_dl: torch.utils.data.DataLoader = dm.val_dataloader()
        val_batch: dict[str, torch.Tensor] = next(iter(val_dl))
        assert val_batch['pixel_values'].shape[0] == config.batch_size


def make_test_model_config() -> GanModelConfig:
    """Creates a small GanModelConfig suitable for CPU testing.

    Uses image_size=64 with num_blocks=4 so spatial progression is:
        encoder: 64 -> 32 -> 16 -> 8 -> 4 -> 1
        decoder: 1 -> 4 -> 8 -> 16 -> 32 -> 64
    """
    return GanModelConfig(
        image_size=TEST_IMAGE_SIZE,
        image_channels=NUM_CHANNELS,
        latent_dim=32,
        base_filters=8,
        num_blocks=4,
        num_smooth_blocks=2,
    )


class TestGanModelConfig:

    def test_default_values(self) -> None:
        """Verifies default GanModelConfig field values."""
        config: GanModelConfig = GanModelConfig()
        assert config.image_size == 256
        assert config.image_channels == 3
        assert config.latent_dim == 128
        assert config.base_filters == 64
        assert config.num_blocks == 5
        assert config.num_smooth_blocks == 1
        assert config.kernel_size == 4
        assert config.stride == 2
        assert config.padding == 1
        assert config.encoder_leaky_relu_slope == 0.2

    def test_frozen(self) -> None:
        """Verifies that GanModelConfig is immutable."""
        config: GanModelConfig = GanModelConfig()
        with pytest.raises(AttributeError):
            config.image_size = 128

    def test_encoder_channels(self) -> None:
        """Verifies per-block encoder channel counts double at each layer."""
        config: GanModelConfig = make_test_model_config()
        channels: list[int] = config.encoder_channels()
        assert channels == [8, 16, 32, 64]

    def test_decoder_channels(self) -> None:
        """Verifies per-block decoder channel counts mirror encoder in reverse."""
        config: GanModelConfig = make_test_model_config()
        channels: list[int] = config.decoder_channels()
        assert channels == [64, 32, 16, 8]

    def test_initial_spatial_size(self) -> None:
        """Verifies the bottleneck spatial size is image_size / 2^num_blocks."""
        config: GanModelConfig = make_test_model_config()
        # 64 / 2^4 = 4
        assert config.initial_spatial_size() == 4


class TestGanModelEncoder:

    def test_forward_shape(self) -> None:
        """Verifies the encoder produces (B, 1) logits from image input."""
        config: GanModelConfig = make_test_model_config()
        encoder: GanModelEncoder = GanModelEncoder(config)

        batch_size: int = 4
        x: torch.Tensor = torch.randn(batch_size, config.image_channels, config.image_size, config.image_size)

        logits: torch.Tensor = encoder(x)
        assert logits.shape == torch.Size([batch_size, 1])

    def test_output_is_unbounded(self) -> None:
        """Verifies the encoder output is raw logits (not clamped to [0, 1])."""
        config: GanModelConfig = make_test_model_config()
        encoder: GanModelEncoder = GanModelEncoder(config)

        batch_size: int = 8
        x: torch.Tensor = torch.randn(batch_size, config.image_channels, config.image_size, config.image_size)

        with torch.no_grad():
            logits: torch.Tensor = encoder(x)

        # raw logits can be positive or negative; just check they are finite
        assert torch.isfinite(logits).all()

    def test_gradient_flows(self) -> None:
        """Verifies gradients propagate through the encoder."""
        config: GanModelConfig = make_test_model_config()
        encoder: GanModelEncoder = GanModelEncoder(config)

        batch_size: int = 2
        x: torch.Tensor = torch.randn(batch_size, config.image_channels, config.image_size, config.image_size)

        logits: torch.Tensor = encoder(x)
        loss: torch.Tensor = logits.mean()
        loss.backward()

        # check at least the first conv layer has gradients
        first_param: torch.nn.Parameter = next(encoder.parameters())
        assert first_param.grad is not None
        assert first_param.grad.shape == first_param.shape


class TestGanModelDecoder:

    def test_forward_shape(self) -> None:
        """Verifies the decoder produces images of correct shape from latent vectors."""
        config: GanModelConfig = make_test_model_config()
        decoder: GanModelDecoder = GanModelDecoder(config)

        batch_size: int = 4
        z: torch.Tensor = torch.randn(batch_size, config.latent_dim)

        images: torch.Tensor = decoder(z)
        assert images.shape == torch.Size([batch_size, config.image_channels, config.image_size, config.image_size])

    def test_output_range(self) -> None:
        """Verifies the decoder output is in [-1, 1] due to Tanh activation."""
        config: GanModelConfig = make_test_model_config()
        decoder: GanModelDecoder = GanModelDecoder(config)

        batch_size: int = 8
        z: torch.Tensor = torch.randn(batch_size, config.latent_dim)

        with torch.no_grad():
            images: torch.Tensor = decoder(z)

        assert images.min().item() >= -1.0
        assert images.max().item() <= 1.0

    def test_gradient_flows(self) -> None:
        """Verifies gradients propagate through the decoder."""
        config: GanModelConfig = make_test_model_config()
        decoder: GanModelDecoder = GanModelDecoder(config)

        batch_size: int = 2
        z: torch.Tensor = torch.randn(batch_size, config.latent_dim)

        images: torch.Tensor = decoder(z)
        loss: torch.Tensor = images.mean()
        loss.backward()

        # check at least the first projection layer has gradients
        first_param: torch.nn.Parameter = next(decoder.parameters())
        assert first_param.grad is not None
        assert first_param.grad.shape == first_param.shape

    def test_different_noise_produces_different_images(self) -> None:
        """Verifies different latent vectors produce different output images."""
        config: GanModelConfig = make_test_model_config()
        decoder: GanModelDecoder = GanModelDecoder(config)
        decoder.eval()

        z1: torch.Tensor = torch.randn(1, config.latent_dim)
        z2: torch.Tensor = torch.randn(1, config.latent_dim)

        with torch.no_grad():
            img1: torch.Tensor = decoder(z1)
            img2: torch.Tensor = decoder(z2)

        # different inputs should produce different outputs
        assert not torch.allclose(img1, img2)


class TestEncoderDecoderRoundTrip:

    def test_encoder_accepts_decoder_output(self) -> None:
        """Verifies the encoder can process images produced by the decoder."""
        config: GanModelConfig = make_test_model_config()
        encoder: GanModelEncoder = GanModelEncoder(config)
        decoder: GanModelDecoder = GanModelDecoder(config)

        batch_size: int = 4
        z: torch.Tensor = torch.randn(batch_size, config.latent_dim)

        with torch.no_grad():
            fake_images: torch.Tensor = decoder(z)
            logits: torch.Tensor = encoder(fake_images)

        assert logits.shape == torch.Size([batch_size, 1])


class TestGanModule:

    def test_forward_generates_images(self) -> None:
        """Verifies forward() generates images from noise via the decoder."""
        config: GanModelConfig = make_test_model_config()
        encoder: GanModelEncoder = GanModelEncoder(config)
        decoder: GanModelDecoder = GanModelDecoder(config)
        module: GanModule = GanModule(config=config, encoder=encoder, decoder=decoder)
        module.eval()

        batch_size: int = 4
        z: torch.Tensor = torch.randn(batch_size, config.latent_dim)

        with torch.no_grad():
            images: torch.Tensor = module(z)

        assert images.shape == torch.Size([batch_size, config.image_channels, config.image_size, config.image_size])

    def test_training_step_discriminator_and_generator(self) -> None:
        """Verifies the GAN training loop (discriminator + generator) produces finite losses."""
        config: GanModelConfig = make_test_model_config()
        encoder: GanModelEncoder = GanModelEncoder(config)
        decoder: GanModelDecoder = GanModelDecoder(config)
        module: GanModule = GanModule(config=config, encoder=encoder, decoder=decoder)

        batch_size: int = 4
        real_images: torch.Tensor = torch.randn(
            batch_size, config.image_channels, config.image_size, config.image_size,
        )

        # manually run the GAN training logic outside of Lightning trainer
        optimizer_d: torch.optim.Adam = torch.optim.Adam(encoder.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optimizer_g: torch.optim.Adam = torch.optim.Adam(decoder.parameters(), lr=2e-4, betas=(0.5, 0.999))

        real_labels: torch.Tensor = torch.full((batch_size, 1), module.REAL_LABEL_SMOOTHING)
        fake_labels: torch.Tensor = torch.zeros(batch_size, 1)
        loss_fn: torch.nn.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()

        # discriminator step
        real_logits: torch.Tensor = encoder(real_images)
        assert real_logits.shape == torch.Size([batch_size, 1])

        noise: torch.Tensor = torch.randn(batch_size, config.latent_dim)
        fake_images: torch.Tensor = decoder(noise)
        fake_logits: torch.Tensor = encoder(fake_images.detach())

        loss_d: torch.Tensor = loss_fn(real_logits, real_labels) + loss_fn(fake_logits, fake_labels)

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # generator step
        fake_logits_g: torch.Tensor = encoder(fake_images)
        loss_g: torch.Tensor = loss_fn(fake_logits_g, real_labels)

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        # verify losses are finite scalars
        assert loss_d.shape == torch.Size([])
        assert loss_g.shape == torch.Size([])
        assert torch.isfinite(loss_d)
        assert torch.isfinite(loss_g)

    def test_configure_optimizers_returns_two_optimizers(self) -> None:
        """Verifies configure_optimizers returns separate optimizers for D and G."""
        config: GanModelConfig = make_test_model_config()
        encoder: GanModelEncoder = GanModelEncoder(config)
        decoder: GanModelDecoder = GanModelDecoder(config)
        module: GanModule = GanModule(config=config, encoder=encoder, decoder=decoder)

        optimizers, schedulers = module.configure_optimizers()

        assert len(optimizers) == 2
        assert len(schedulers) == 0
        assert isinstance(optimizers[0], torch.optim.Adam)
        assert isinstance(optimizers[1], torch.optim.Adam)

    def test_automatic_optimization_is_disabled(self) -> None:
        """Verifies that manual optimization is used (required for GAN training)."""
        config: GanModelConfig = make_test_model_config()
        encoder: GanModelEncoder = GanModelEncoder(config)
        decoder: GanModelDecoder = GanModelDecoder(config)
        module: GanModule = GanModule(config=config, encoder=encoder, decoder=decoder)

        assert module.automatic_optimization is False
