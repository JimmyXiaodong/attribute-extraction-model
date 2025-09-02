from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.base_model.definition.const import CONST


def get_tokenizer_and_sequence_classification_model(
        pretrained_model_path,
        num_labels=2
) -> (AutoTokenizer, AutoModelForSequenceClassification):
    """
    get tokenizer and sequence_classification_model from pretrained
    language model
    :param pretrained_model_path: pretrained language model path
    :param num_labels: the number of prediction labels, default is 2
    :return: tokenizer and model
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_path, num_labels=num_labels)
    return tokenizer, model


def get_encoder_outputs(
        cls,
        encoder,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None):
    """
    Get encoder model outputs.
    """
    if cls.base_model_prefix == CONST.MODEL_TYPE.NEZHA:
        outputs = encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in [
                CONST.POOLER_TYPE.AVG_TOP2,
                CONST.POOLER_TYPE.AVG_FIRST_LAST] else False,
            return_dict=True,
        )
    else:
        outputs = encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in [
                CONST.POOLER_TYPE.AVG_TOP2,
                CONST.POOLER_TYPE.AVG_FIRST_LAST] else False,
            return_dict=True,
        )
    return outputs
