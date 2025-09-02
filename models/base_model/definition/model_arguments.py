from dataclasses import dataclass, field
from typing import Optional

from transformers import MODEL_FOR_MASKED_LM_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class BaseModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to
    fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    cache_dir: Optional[str] = field(
        default="./pretrained_models",
        metadata={
            "help": "Where do you want to store the pretrained models "
                    "downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the "
                    "tokenizers library) or not."},
    )


@dataclass
class BaseDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for
    training and evaluation.
    """

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set "
                    "in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."},
    )

    data_cache_dir: Optional[str] = field(
        default="./data/cache",
        metadata={
            "help": "Where do you want to store the preprocessed data."},
    )

    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The training data file (.txt or .csv)."}
    )

    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "The validation data file (.txt or .csv)."}
    )

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after "
                    "tokenization. Sequences longer than this will be "
                    "truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when "
                    "batching to the maximum length in the batch."
        },
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv", "json", "txt"
                ], "`train_file` should be a csv, a json or a txt file."
