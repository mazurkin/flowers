import collections
import dataclasses
import logging
import typing as t

import PIL.Image
import albumentations
import datasets
import numpy

import torch
import torch.utils

from lightning import pytorch as pl


@dataclasses.dataclass(frozen=True)
class GanDataModuleConfig:
    """Configuration for GanDataModule."""

    # target image height in pixels
    image_height: int = dataclasses.field(
        default=224,
        metadata={'help': 'Target image height in pixels'},
    )

    # target image width in pixels
    image_width: int = dataclasses.field(
        default=224,
        metadata={'help': 'Target image width in pixels'},
    )

    # batch size for training
    batch_size: int = dataclasses.field(
        default=64,
        metadata={'help': 'Batch size for data loaders'},
    )

    # number of data loading workers
    num_workers: int = dataclasses.field(
        default=4,
        metadata={'help': 'Number of data loading workers'},
    )

    # fraction of data used for training
    train_fraction: float = dataclasses.field(
        default=0.8,
        metadata={'help': 'Fraction of dataset used for training'},
    )

    # fraction of data used for evaluation (validation)
    eval_fraction: float = dataclasses.field(
        default=0.1,
        metadata={'help': 'Fraction of dataset used for evaluation'},
    )

    # random seed for reproducible dataset splitting
    seed: int = dataclasses.field(
        default=42,
        metadata={'help': 'Random seed for dataset splitting'},
    )

    # number of largest classes to keep (None means keep all classes)
    top_classes: t.Optional[int] = dataclasses.field(
        default=5,
        metadata={'help': 'Number of largest classes to keep (None = all)'},
    )

    # ImageNet channel means for normalization (R, G, B)
    IMAGENET_MEAN: t.ClassVar[tuple[float, float, float]] = (0.485, 0.456, 0.406)

    # ImageNet channel standard deviations for normalization (R, G, B)
    IMAGENET_STD: t.ClassVar[tuple[float, float, float]] = (0.229, 0.224, 0.225)


class GanDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace dataset split and applies albumentations transforms.

    Each sample is a dict with:
      - 'image': PIL.Image from the HuggingFace dataset
      - 'label': int flower category index (0 to 101)

    Parameters:
        hf_dataset: a HuggingFace dataset split containing 'image' and 'label' columns
        transform: an albumentations Compose pipeline to apply to each image
    """

    # number of channels expected in the output image (RGB)
    NUM_CHANNELS: t.Final[int] = 3

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        transform: albumentations.Compose,
    ) -> None:
        self.hf_dataset: t.Final[datasets.Dataset] = hf_dataset
        self.transform: t.Final[albumentations.Compose] = transform

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Returns a single sample as a dict with 'pixel_values' tensor and 'label' tensor.

        Parameters:
            index: the index of the sample to retrieve

        Returns:
            dict with 'pixel_values' (C, H, W) float tensor and 'label' scalar long tensor
        """
        sample: dict = self.hf_dataset[index]

        # convert PIL image to RGB to handle grayscale or RGBA inputs
        image: PIL.Image.Image = sample['image'].convert('RGB')

        # convert to numpy array (H, W, C) with uint8 values for albumentations
        image_np: numpy.ndarray = numpy.array(image)

        # apply albumentations pipeline, result contains 'image' key as a torch tensor
        transformed: dict = self.transform(image=image_np)

        pixel_values: torch.Tensor = transformed['image']
        assert pixel_values.shape[0] == self.NUM_CHANNELS, (
            f'expected {self.NUM_CHANNELS} channels, got {pixel_values.shape[0]}'
        )

        label: torch.Tensor = torch.tensor(sample['label'], dtype=torch.long)

        return {
            'pixel_values': pixel_values,
            'label': label,
        }


class GanDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the mteb/oxford-flowers dataset.

    Loads the dataset from HuggingFace, splits it into train and eval sets,
    and applies albumentations image transforms including augmentation for training.

    Parameters:
        config: GanDataModuleConfig with image size, batch size, split fractions, etc.
    """

    # the HuggingFace dataset identifier
    DATASET_NAME: t.Final[str] = 'mteb/oxford-flowers'

    def __init__(self, config: GanDataModuleConfig) -> None:
        super().__init__()

        self.logging: t.Final[logging.Logger] = logging.getLogger(self.__class__.__name__)

        self.config: t.Final[GanDataModuleConfig] = config

        # datasets will be assigned after setup()
        self.train_dataset: t.Optional[GanDataset] = None
        self.eval_dataset: t.Optional[GanDataset] = None

    def build_train_transform(self) -> albumentations.Compose:
        """Builds the albumentations augmentation pipeline for the training set.

        Includes random resized crop, rotation, horizontal flip, color jitter, and normalization.

        Returns:
            albumentations.Compose pipeline producing (C, H, W) float tensors
        """
        return albumentations.Compose([
            albumentations.RandomResizedCrop(
                size=(self.config.image_height, self.config.image_width),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            albumentations.Rotate(
                limit=10,
                p=0.5,
            ),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.8,
            ),
            albumentations.GaussianBlur(
                blur_limit=(3, 7),
                p=0.1,
            ),
            albumentations.Normalize(
                mean=GanDataModuleConfig.IMAGENET_MEAN,
                std=GanDataModuleConfig.IMAGENET_STD,
            ),
            albumentations.pytorch.ToTensorV2(),
        ])

    def build_eval_transform(self) -> albumentations.Compose:
        """Builds the albumentations transform pipeline for the evaluation set.

        Only resizes and normalizes, no augmentation.

        Returns:
            albumentations.Compose pipeline producing (C, H, W) float tensors
        """
        return albumentations.Compose([
            albumentations.Resize(
                height=self.config.image_height,
                width=self.config.image_width,
            ),
            albumentations.Normalize(
                mean=GanDataModuleConfig.IMAGENET_MEAN,
                std=GanDataModuleConfig.IMAGENET_STD,
            ),
            albumentations.pytorch.ToTensorV2(),
        ])

    @t.override
    def setup(self, stage: t.Optional[str] = None) -> None:
        """Loads and splits the HuggingFace dataset, then wraps each split with transforms.

        Parameters:
            stage: the Lightning stage ('fit', 'validate', or None for all)
        """
        # load the full dataset (only a 'train' split exists)
        self.logging.info('loading dataset: %s', self.DATASET_NAME)
        full_dataset: datasets.Dataset = datasets.load_dataset(
            self.DATASET_NAME,
            split='train',
        )
        self.logging.info('total samples: %d', len(full_dataset))

        # filter to the top N largest classes if configured
        if self.config.top_classes is not None:
            label_counts: collections.Counter = collections.Counter(full_dataset['label'])

            top_labels: set[int] = {
                label for label, _count in label_counts.most_common(self.config.top_classes)
            }

            full_dataset = full_dataset.filter(lambda row: row['label'] in top_labels)

            self.logging.info(
                'filtered to top %d classes (labels=%s): %d samples',
                self.config.top_classes, sorted(top_labels), len(full_dataset),
            )

        # split into train and eval sets
        split: datasets.DatasetDict = full_dataset.train_test_split(
            test_size=self.config.eval_fraction,
            seed=self.config.seed,
            shuffle=True,
        )

        train_hf: datasets.Dataset = split['train']
        eval_hf: datasets.Dataset = split['test']

        self.logging.info('train samples: %d, eval samples: %d', len(train_hf), len(eval_hf))

        # build transforms
        train_transform: albumentations.Compose = self.build_train_transform()
        eval_transform: albumentations.Compose = self.build_eval_transform()

        # wrap into GanDataset instances
        if stage in ('fit', None):
            self.train_dataset = GanDataset(train_hf, train_transform)
            self.eval_dataset = GanDataset(eval_hf, eval_transform)

        if stage in ('validate', None):
            self.eval_dataset = GanDataset(eval_hf, eval_transform)

    @t.override
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the training DataLoader with shuffling enabled.

        Returns:
            DataLoader over the training split
        """
        assert self.train_dataset is not None, 'call setup() before train_dataloader()'

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    @t.override
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the evaluation (validation) DataLoader.

        Returns:
            DataLoader over the evaluation split
        """
        assert self.eval_dataset is not None, 'call setup() before val_dataloader()'

        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )


