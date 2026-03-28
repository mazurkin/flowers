import argh
import pathlib
import logging
import logging.config
import typing as t
import yaml
import sys
import json
import warnings

import torch
import torch.cuda
import torch.nn
import torch.profiler
import torch.multiprocessing
import torch.distributed

import ml.module


# noinspection DuplicatedCode,PyMethodMayBeStatic
class TrainerApplication:
    """
    model builder application
    """

    PATH_APPLICATION: t.Final[pathlib.Path] = pathlib.Path(__file__)

    PATH_DIR_SOURCES: t.Final[pathlib.Path] = PATH_APPLICATION.parent.resolve()

    PATH_DIR_PACKAGE: t.Final[pathlib.Path] = PATH_DIR_SOURCES.parent.resolve()

    PATH_DIR_WORK: t.Final[pathlib.Path] = PATH_DIR_PACKAGE / 'work'

    def __init__(self):
        # initialize logging
        logging_config_path: pathlib.Path = self.PATH_DIR_SOURCES / 'trainer.yaml'
        logging_config = self.load_yaml(logging_config_path, yaml.SafeLoader)
        logging.config.dictConfig(logging_config)
        logging.info('using logging configuration [%s]', logging_config_path)

        # local logger
        self.logger = logging.getLogger('application')
        self.logger.info('command line       :\n%s', json.dumps(sys.argv[1:], default=str, indent=2, sort_keys=False))
        self.logger.info('torch start        : %s', torch.multiprocessing.get_start_method())

        # avoid the warning:
        # TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.
        # Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
        torch.set_float32_matmul_precision('high')

        # default device is CPU
        torch.set_default_device('cpu')

        # default dtype
        torch.set_default_dtype(torch.float32)

        # torch multi-threading setup
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # determinism
        torch.use_deterministic_algorithms(mode=False)

        # avoid the CheckPoint warning
        warnings.filterwarnings(
            action='ignore',
            message=r'Checkpoint directory .* exists and is not empty\.',
            category=UserWarning,
        )
        warnings.filterwarnings(
            action='ignore',
            message=r'.* is set, but there is no last checkpoint available\. No checkpoint will be loaded\. .*',
            category=UserWarning,
        )

        # avoid the LitLogger warning
        warnings.filterwarnings(
            action='ignore',
            message=r'LitLogger does not support `log_graph`',
            category=UserWarning,
        )

    def train(self):
        trainer: ml.module.GanTrainer = ml.module.GanTrainer(
            work_folder_path=self.PATH_DIR_WORK,
        )

        trainer.run()

    @staticmethod
    def load_yaml(path: pathlib.Path, yaml_loader_class: t.Type) -> t.Dict:
        with path.open('rt') as file:
            yaml_text = file.read()

        # noinspection PyTypeChecker
        yaml_dict = yaml.load(yaml_text, yaml_loader_class)

        return yaml_dict


if __name__ == '__main__':
    application = TrainerApplication()

    parser = argh.ArghParser()
    argh.add_commands(parser, [application.train])

    try:
        argh.dispatch(parser)
    finally:
        logging.info('the work is finished')
        logging.shutdown()
