import datetime
import logging
import pathlib
import typing as t

import torch
import torch.nn
import torch.optim
import torch.utils.data

import torchvision.utils

import lightning.pytorch as pl
import lightning.pytorch.loggers
import lightning.pytorch.callbacks

from ml.data import FlowersDataModuleConfig, FlowersDataModule
from ml.model import FlowersModelConfig, FlowersModelEncoder, FlowersModelDecoder


# noinspection DuplicatedCode,PyMethodMayBeStatic
class FlowersModule(pl.LightningModule):
    """PyTorch Lightning module for GAN training on flower images.

    Wraps the encoder (discriminator) and decoder (generator) with standard
    GAN adversarial loss (BCEWithLogitsLoss). Uses manual optimization with
    separate Adam optimizers for generator and discriminator.

    The training_step alternates between:
      1) discriminator update: maximize log(D(real)) + log(1 - D(G(z)))
      2) generator update: maximize log(D(G(z)))

    Parameters:
        config: FlowersModelConfig shared between encoder and decoder
        encoder: FlowersModelEncoder (discriminator)
        decoder: FlowersModelDecoder (generator)
        learning_rate_d: learning rate for the discriminator (encoder) optimizer
        learning_rate_g: learning rate for the generator (decoder) optimizer
        beta1: Adam beta1 parameter (low value per DCGAN convention)
        beta2: Adam beta2 parameter
    """

    # label smoothing value for real labels to stabilize discriminator training
    REAL_LABEL_SMOOTHING: t.Final[float] = 0.9

    # weight for spatial variance regularization (encourages contrast and texture)
    SPATIAL_VARIANCE_WEIGHT: t.Final[float] = 0.5

    # weight for channel diversity regularization (encourages color, penalizes gray)
    CHANNEL_DIVERSITY_WEIGHT: t.Final[float] = 0.5

    # number of sample images to generate for TensorBoard visualization
    NUM_SAMPLE_IMAGES: t.Final[int] = 16

    # number of images per row in the TensorBoard grid
    GRID_NROW: t.Final[int] = 4

    # DCGAN weight initialization: normal distribution with mean=0, stdev=0.02
    INIT_WEIGHT_STD: t.Final[float] = 0.02

    # batch normalization initialization: mean=1, stdev=0.02
    INIT_BN_MEAN: t.Final[float] = 1.0

    def __init__(
        self,
        config: FlowersModelConfig,
        encoder: FlowersModelEncoder,
        decoder: FlowersModelDecoder,
        learning_rate_d: float = 2e-4,
        learning_rate_g: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            ignore=['config', 'encoder', 'decoder'],
        )

        # GAN uses manual optimization for separate generator/discriminator steps
        self.automatic_optimization = False

        self.config: t.Final[FlowersModelConfig] = config

        # optimizer hyperparameters (asymmetric LR: lower D rate prevents D from overpowering G)
        self.learning_rate_d: t.Final[float] = learning_rate_d
        self.learning_rate_g: t.Final[float] = learning_rate_g
        self.beta1: t.Final[float] = beta1
        self.beta2: t.Final[float] = beta2

        # discriminator (encoder) and generator (decoder)
        self.encoder: t.Final[FlowersModelEncoder] = encoder
        self.decoder: t.Final[FlowersModelDecoder] = decoder

        # binary cross-entropy with logits for real/fake classification
        self.loss_fn: t.Final[torch.nn.BCEWithLogitsLoss] = torch.nn.BCEWithLogitsLoss()

        # fixed noise vector for consistent sample visualization across epochs
        self.fixed_noise: t.Final[torch.Tensor] = torch.randn(self.NUM_SAMPLE_IMAGES, config.latent_dim)

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module) -> None:
        """Initializes Conv2d, ConvTranspose2d, Linear and BatchNorm layers per DCGAN convention.

        All convolutional and linear weights are drawn from Normal(0, 0.02). BatchNorm weights
        are drawn from Normal(1, 0.02) with biases zeroed.

        Parameters:
            module: a single layer to initialize (called via self.apply)
        """
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.INIT_WEIGHT_STD)
        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            torch.nn.init.normal_(module.weight, mean=self.INIT_BN_MEAN, std=self.INIT_WEIGHT_STD)
            torch.nn.init.zeros_(module.bias)

    @t.override
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generates images from latent noise vectors (generator forward pass).

        Parameters:
            z: latent noise vector of shape (B, latent_dim)

        Returns:
            generated images of shape (B, image_channels, image_size, image_size)
        """
        return self.decoder(z)

    @t.override
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Performs one GAN training step: discriminator update then generator update.

        Parameters:
            batch: dict with 'pixel_values' (B, C, H, W) tensor
            batch_idx: index of the current batch
        """
        optimizer_d, optimizer_g = self.optimizers()

        real_images: torch.Tensor = batch['pixel_values']
        batch_size: int = real_images.shape[0]

        # labels for real and fake samples
        real_labels: torch.Tensor = torch.full(
            (batch_size, 1), self.REAL_LABEL_SMOOTHING,
            device=self.device, dtype=real_images.dtype,
        )
        fake_labels: torch.Tensor = torch.zeros(
            batch_size, 1,
            device=self.device, dtype=real_images.dtype,
        )

        # ------------------------------------------------------------------
        # discriminator step: maximize log(D(real)) + log(1 - D(fake))
        # ------------------------------------------------------------------

        # discriminator on real images
        real_logits: torch.Tensor = self.encoder(real_images)
        assert real_logits.shape == torch.Size([batch_size, 1])

        loss_d_real: torch.Tensor = self.loss_fn(real_logits, real_labels)

        # generate fake images
        noise: torch.Tensor = torch.randn(
            batch_size,
            self.config.latent_dim,
            device=self.device,
            dtype=real_images.dtype,
        )

        fake_images: torch.Tensor = self.decoder(noise)
        assert fake_images.shape == real_images.shape

        # discriminator on fake images (detach to avoid backprop through generator)
        fake_logits: torch.Tensor = self.encoder(fake_images.detach())
        assert fake_logits.shape == torch.Size([batch_size, 1])

        loss_d_fake: torch.Tensor = self.loss_fn(fake_logits, fake_labels)

        # total discriminator loss
        loss_d: torch.Tensor = loss_d_real + loss_d_fake

        optimizer_d.zero_grad()
        self.manual_backward(loss_d)
        optimizer_d.step()

        # ------------------------------------------------------------------
        # generator step: maximize log(D(G(z)))
        # ------------------------------------------------------------------

        # re-evaluate discriminator on fake images (now allow gradients to flow to generator)
        fake_logits_g: torch.Tensor = self.encoder(fake_images)
        assert fake_logits_g.shape == torch.Size([batch_size, 1])

        # generator wants discriminator to classify fakes as real
        loss_g_adv: torch.Tensor = self.loss_fn(fake_logits_g, real_labels)

        # ------------------------------------------------------------------
        # vibrancy regularization: penalize dim/gray outputs
        # ------------------------------------------------------------------

        # spatial variance: encourage high contrast and texture across pixels
        # compute per-image std across spatial dims (H, W), then average over batch and channels
        spatial_std: torch.Tensor = fake_images.std(dim=(-2, -1)).mean()
        assert spatial_std.shape == torch.Size([])

        loss_spatial: torch.Tensor = -spatial_std

        # channel diversity: encourage different R, G, B values (penalize gray)
        # compute per-image mean per channel, then std across channels, average over batch
        channel_means: torch.Tensor = fake_images.mean(dim=(-2, -1))
        assert channel_means.shape == torch.Size([batch_size, self.config.image_channels])

        channel_std: torch.Tensor = channel_means.std(dim=-1).mean()
        assert channel_std.shape == torch.Size([])

        loss_channel: torch.Tensor = -channel_std

        # total generator loss
        loss_g: torch.Tensor = loss_g_adv \
            + self.SPATIAL_VARIANCE_WEIGHT * loss_spatial \
            + self.CHANNEL_DIVERSITY_WEIGHT * loss_channel

        optimizer_g.zero_grad()
        self.manual_backward(loss_g)
        optimizer_g.step()

        # ------------------------------------------------------------------
        # logging
        # ------------------------------------------------------------------

        self.log('train/loss_d', loss_d, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/loss_g', loss_g, prog_bar=True, on_step=False, on_epoch=True)

        self.log('train/loss_g_adv', loss_g_adv, on_step=False, on_epoch=True)
        self.log('train/loss_spatial', loss_spatial, on_step=False, on_epoch=True)
        self.log('train/loss_channel', loss_channel, on_step=False, on_epoch=True)

        self.log('train/loss_d_real', loss_d_real, on_step=False, on_epoch=True)
        self.log('train/loss_d_fake', loss_d_fake, on_step=False, on_epoch=True)

        # discriminator confidence on real and fake (sigmoid of mean logit)
        self.log('train/d_real', torch.sigmoid(real_logits).mean(), on_step=False, on_epoch=True)
        self.log('train/d_fake', torch.sigmoid(fake_logits).mean(), on_step=False, on_epoch=True)

    @t.override
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Computes validation losses for discriminator and generator.

        Parameters:
            batch: dict with 'pixel_values' (B, C, H, W) tensor
            batch_idx: index of the current batch
        """
        real_images: torch.Tensor = batch['pixel_values']
        batch_size: int = real_images.shape[0]

        real_labels: torch.Tensor = torch.ones(
            batch_size, 1,
            device=self.device, dtype=real_images.dtype,
        )
        fake_labels: torch.Tensor = torch.zeros(
            batch_size, 1,
            device=self.device, dtype=real_images.dtype,
        )

        # discriminator on real images
        real_logits: torch.Tensor = self.encoder(real_images)
        loss_d_real: torch.Tensor = self.loss_fn(real_logits, real_labels)

        # generate fake images
        noise: torch.Tensor = torch.randn(
            batch_size, self.config.latent_dim,
            device=self.device, dtype=real_images.dtype,
        )
        fake_images: torch.Tensor = self.decoder(noise)

        # discriminator on fake images
        fake_logits: torch.Tensor = self.encoder(fake_images)
        loss_d_fake: torch.Tensor = self.loss_fn(fake_logits, fake_labels)

        loss_d: torch.Tensor = loss_d_real + loss_d_fake
        loss_g: torch.Tensor = self.loss_fn(fake_logits, real_labels)

        self.log('val/loss_d', loss_d, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/loss_g', loss_g, prog_bar=True, on_step=False, on_epoch=True)

        self.log('val/d_real', torch.sigmoid(real_logits).mean(), on_step=False, on_epoch=True)
        self.log('val/d_fake', torch.sigmoid(fake_logits).mean(), on_step=False, on_epoch=True)

    @t.override
    def on_validation_epoch_end(self) -> None:
        """Generates sample images from fixed noise and logs them to TensorBoard."""
        # move fixed noise to the current device
        z: torch.Tensor = self.fixed_noise.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            fake_images: torch.Tensor = self.decoder(z)

        # decoder outputs [-1, 1] via Tanh, rescale to [0, 1] for visualization
        fake_images = (fake_images + 1.0) / 2.0
        fake_images = fake_images.clamp(0.0, 1.0)

        # make a grid of generated images
        grid: torch.Tensor = torchvision.utils.make_grid(
            fake_images,
            nrow=self.GRID_NROW,
            normalize=False,
        )

        # log to the trainer's TensorBoard logger
        tensorboard: t.Optional[lightning.pytorch.loggers.TensorBoardLogger] = None
        for logger in self.loggers:
            if isinstance(logger, lightning.pytorch.loggers.TensorBoardLogger):
                tensorboard = logger
                break

        if tensorboard is not None:
            tensorboard.experiment.add_image(
                'generated_samples',
                grid,
                global_step=self.current_epoch,
            )

    @t.override
    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list]:
        """Configures separate Adam optimizers for discriminator and generator.

        Returns:
            tuple of (optimizers_list, empty_schedulers_list)
        """
        optimizer_d: torch.optim.Adam = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.learning_rate_d,
            betas=(self.beta1, self.beta2),
        )

        optimizer_g: torch.optim.Adam = torch.optim.Adam(
            self.decoder.parameters(),
            lr=self.learning_rate_g,
            betas=(self.beta1, self.beta2),
        )

        return [optimizer_d, optimizer_g], []


# noinspection DuplicatedCode,PyMethodMayBeStatic
class FlowersTrainer:
    """Assembles the GAN model, data module, Lightning module, and trainer.

    Parameters:
        work_folder_path: directory for checkpoints, logs, and generated samples
    """

    def __init__(self, work_folder_path: pathlib.Path):
        # work folder
        self.work_folder_path: t.Final[pathlib.Path] = work_folder_path

        # ----------------------------------------------------------------------
        # model config and networks
        # ----------------------------------------------------------------------

        self.model_config: t.Final[FlowersModelConfig] = FlowersModelConfig()

        self.encoder: t.Final[FlowersModelEncoder] = FlowersModelEncoder(
            config=self.model_config,
        )

        self.decoder: t.Final[FlowersModelDecoder] = FlowersModelDecoder(
            config=self.model_config,
        )

        # ----------------------------------------------------------------------
        # data module (image size must match the model's image_size)
        # ----------------------------------------------------------------------

        self.data_module_config: t.Final[FlowersDataModuleConfig] = FlowersDataModuleConfig(
            image_height=self.model_config.image_size,
            image_width=self.model_config.image_size,
        )

        self.data_module: t.Final[FlowersDataModule] = FlowersDataModule(
            config=self.data_module_config,
        )

        # ----------------------------------------------------------------------
        # Lightning GAN module
        # ----------------------------------------------------------------------

        self.model_module: t.Final[FlowersModule] = FlowersModule(
            config=self.model_config,
            encoder=self.encoder,
            decoder=self.decoder,
        )

        # ----------------------------------------------------------------------
        # loggers
        # ----------------------------------------------------------------------

        self.tensorboard_logger: t.Final[lightning.pytorch.loggers.Logger] = \
            lightning.pytorch.loggers.TensorBoardLogger(
                save_dir=self.work_folder_path / 'tensorboard',
                name='',
                version='',
                default_hp_metric=False,
            )

        # ----------------------------------------------------------------------
        # callbacks
        # ----------------------------------------------------------------------

        self.learning_rate_callback: t.Final[lightning.pytorch.callbacks.LearningRateMonitor] = \
            lightning.pytorch.callbacks.LearningRateMonitor(
                logging_interval='epoch',
                log_momentum=True,
                log_weight_decay=True,
            )

        self.progress_callback: t.Final[lightning.pytorch.callbacks.TQDMProgressBar] = \
            lightning.pytorch.callbacks.TQDMProgressBar(
                leave=True,
            )

        self.timer_callback: t.Final[lightning.pytorch.callbacks.Timer] = \
            lightning.pytorch.callbacks.Timer(
                duration=datetime.timedelta(hours=4),
            )

        self.last_checkpoint_callback: t.Final[pl.callbacks.ModelCheckpoint] = \
            lightning.pytorch.callbacks.ModelCheckpoint(
                dirpath=self.work_folder_path / 'snapshot' / 'last',
                filename='epoch-{epoch:03d}',
                save_top_k=1,
                save_last='link',
                save_weights_only=False,
                save_on_train_epoch_end=True,
                auto_insert_metric_name=False,
                verbose=True,
            )

        # ----------------------------------------------------------------------
        # trainer
        # ----------------------------------------------------------------------

        self.trainer: t.Final[pl.Trainer] = pl.Trainer(
            min_epochs=1,
            max_epochs=256,

            min_steps=1,
            max_steps=-1,

            num_sanity_val_steps=2,

            limit_train_batches=None,
            limit_val_batches=None,
            limit_test_batches=None,
            limit_predict_batches=None,

            gradient_clip_val=None,

            enable_model_summary=True,
            enable_autolog_hparams=True,
            enable_progress_bar=True,
            enable_checkpointing=True,

            log_every_n_steps=10,

            accumulate_grad_batches=1,
            use_distributed_sampler=False,

            strategy='auto',
            accelerator='cuda',
            precision='32-true',

            num_nodes=1,
            devices=1,

            logger=[
                self.tensorboard_logger,
            ],

            callbacks=[
                self.learning_rate_callback,
                self.progress_callback,
                self.timer_callback,
                self.last_checkpoint_callback,
            ],
        )

    def run(self) -> None:
        """Runs GAN training."""
        self.trainer.fit(
            model=self.model_module,
            datamodule=self.data_module,
            ckpt_path='last',
        )
