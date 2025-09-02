import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments, MODEL_FOR_MASKED_LM_MAPPING

from models.base_model.definition.model_arguments import (
    BaseModelArguments,
    BaseDataTrainingArguments
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
print(root_path)

@dataclass
class AttributeExtractionModelArguments(BaseModelArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to
    fine-tune, or train from scratch.
    """


@dataclass
class AttributeExtractionModelDataTrainingArguments(BaseDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for
    training and evaluation.
    """

    token_label_file: str = field(
        default=None,
        metadata={
            "help": "The file (.json) of all token labels for our data)"}
    )

    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The training data file (.json)."}
    )

    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "The validation data file (.json)."}
    )

    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "The test data file (.json)."}
    )


@dataclass
class AttributeExtractionModelTrainingArguments(TrainingArguments):
    """
    The parameters used in our model training in addition to the original
    parameters of Huggingface.
    """
    dataset_date: Optional[str] = field(
        default=None,
        metadata={"help": "The creation date of the dataset used in training."}
    )

    metric_path: Optional[str] = field(
        default=os.path.join(root_path, "models/base_model/metrics/definition/seqeval"),
        metadata={"help": "The path to load metrics for model evaluation."}
    )