# project

this the project with uses PyTorch, PyTorch Lightning, Transformers and
libraries to albumentations to build a GAN generator for the flowers

## run

The proper way to run the commands is using the script `bin/run` which activates the environment and sets the Python paths:

```shell
bin/run python --version
```

```shell
bin/run hf version
```

You can run the application itself in the same way:
```shell
bin/run python3 src/builder.py train --help
```

You can also refer to the `Makefile` file in the root of the project if you need to see the sample examples of
how to run the data preprocessing, the model trainer or any other project's tool or command.

## code style

### dataclasses

Where it is possible use dataclasses for the "anemic" beans which contains just a state and no sophisticated logic.
Make them frozen if possible. Do not use the dataclass approach for the classes with heavy complicated logic.

Make a comment for the dataclass bean and for its fields, example:

```python
@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    # name of the HuggingFace model
    descriptor: str = field(
        default='answerdotai/ModernBERT-base',
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )

    # data type for the model
    dtype: str = field(
        default='auto',
        metadata={'help': 'Torch dtype for model (auto, float16, bfloat16, float32)'}
    )
```

### PyTorch operations

After every PyTorch operation add additional `assert` which check the dimension of the result tensor, example:

```python
combined_emb: torch.Tensor = torch.cat([token_emb, dayofweek_emb, batch.embedding_features], dim=-1)
assert torch.Size([batch.size, batch.period_days, self.token_projection_size]) == combined_emb.shape
```

### comments

Add all necessary comments which will help you later to understand the whole architecture and the specific
moments in the code.

Prefer starting the short comments with lower-case.

### python imports

All Python imports must be made in the beginning of the Python file. You cannot put any import in the
middle of the script.

For readability use fully qualified class names, for example prefer `pathlib.Path()` instead of just `Path`.
Accordingly use `import pathlib` instead of `from pathlib import Path`.

For the dataclasses use this example:

```python
import dataclasses

@dataclasses.dataclass
class Bean:
    """Description of the bean's purpose"""

    # the description for the field
    value: dataclasses.field(default=256)
```

The priority is readability and unambiguity, not the shorter code.

### readability

Make everything to make the code simple and readable, make clear names for the variables and methods. All numeric
constants should be named if it makes sense.

### type

For method parameters, for the dataclass fields, for the local variables use types. Where possible use embedded types
like `int`, `str`, `dict[str]`, `tuple[str, int]`, `list[int]`.

Where it is needed you shall use the types from the `typing` module like:

```python
import typing as t

items: t.Iterator[str] = get_items()
```

Use `@override` decorator where it is required:
```python
import typing as t
import pytorch_lightning as pl

class MNISTClassifier(pl.LightningModule):

    @t.override
    def forward(self, x):
        return self.net(x)
```

### immutability

In the `__init__` method use `t.Final` where possible to mark the attribute as immutable, for example:

```python
import pathlib
import typing as t

def __init__(self, path: pathlib.Path):
    # the attribute `self.path` is immutable and will never be modified
    self.path: t.Final[pathlib.Path] = path
```

### constants

Make an immutable constant for any "magic" numeric constant which is not 0, 1 or -1, for example:
```python
import typing as t

class Calculator:

    # the number of seconds in one minute
    SECONDS_IN_MINUTE: t.Final[int] = 60

    def convert_min_to_sec(self, mins: float) -> float:
        return mins * self.SECONDS_IN_MINUTE
```

All constant names must be in UPPER_CASE.

You can make exclusion for initialization of a dataclass, where the name of the parameter is explicitly mentioned:
```python
import typing as t

config: TrainerConfig = TrainerConfig(
    adapter_num_heads=4,
    adapter_num_layers=2,
)
```

### strings

Use single-quotes for the string, not the double-quotes. Convert double-quotes to the single quotes if required.
Use f-strings where it is possible instead of string concatenation with the plus sign.

### list and maps

Always leave the trailing comma in multi-live list and map definitions, it helps to avoid unnecessary changes, example:
```python
items: list[str] = [
    'aaa',
    'bbb',
]
```

Here if the additional third line `'ccc'` is added later, the second line will be untouched as the comma
is already there:
```python
items: list[str] = [
    'aaa',
    'bbb',
    'ccc',
]
```

### private fields

Do not use the leading underscore for the private fields and methods. Use just "value" not "_value".

### line length

Line width should not exceed 120 characters, if it does split the line into multiple lines according
to the Python language rules.

### enums

Prefer enums for the collection of the names tags, use `enum.IntEnum` and `enum.StrEnum` where it is possible,
for example this is a good usage of `enum.StrEnum`:

```python
@enum.unique
class MetricsTag(enum.StrEnum):

    LR = 'lr'

    EPOCH = 'epoch'

    ELAPSED = 'elapsed'

    EVALUATION = 'evaluation'

    LOSS = 'loss'

    SCORE = 'score'
```

### logging

Use logging where it makes sense to produce output which you can analyze later,
use standard Python logging placeholders where it is possible, example:

```python
logging.info('Total parameters: %s', total_params)
```

The following is a wrong example, f-string is NOT a standard way of logging, avoid it:

```python
logging.info(f'Total parameters: {total_params}')
```

## tests

Create and keep the tests in `tst` folder, all tests must run on "cpu" PyTorch device.

To run the tests use the `bin/run` script, example:

```shell
bin/run pytest tst/
```

The environment variable `PACKAGE_DIR` points to the root of this project and is set in the `bin/run` script, you
can use this environment variable to calculate the path to the data folder like this:

```python
TEST_DATA_PATH: pathlib.Path = pathlib.Path(os.environ['PACKAGE_DIR']) / 'data' / 'test'
```

For the tests you can use only the files in `data/test` subfolder as any other subfolders in the `data` folder
are temporary and are not kept in the git repository.

For the ad-hoc jobs like building the model with `src/builder.py` script you can use the whole `data` folder
including all it's subfolders.

## documentation

Add standard Python documentation for every implemented method describing the method's parameters and the result.

## git, version control

You can do commits if the user asked you to with the regular git commands like `git add .` and `git commit -m 'description'`,
access and the remote repository are already set up.

## workflow

Always ask the user if you are not sure what to do, you know a better option, or there are better alternatives
to what the user has suggested. Share important knowledge which could help working on the issue.
