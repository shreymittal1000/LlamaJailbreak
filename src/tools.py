from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, TFPreTrainedModel

def prepare_input_string(sysprompt: str, input_string: str) -> str:
    """
    A function to prepare the input string for the model.
    :param sysprompt: The system prompt to be added to the input string.
    :param input_string: The input string to be prepared.
    :return: The prepared input string.
    """
    return sysprompt + "<<USER>> " + input_string + " <</USER>> <<YOU>> "


def get_sentence_embedding(model: PreTrainedModel | TFPreTrainedModel, tokenizer: PreTrainedTokenizer, sentence: str):
    """
    A function to get the sentence embedding of a sentence.
    :param model: The model to be used.
    :param tokenizer: The tokenizer to be used.
    :param sentence: The sentence to get the embedding of.
    :return: The sentence embedding.
    """
    tokenized = tokenizer(sentence.strip().replace('"', ""), return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    
    word_embeddings = model.get_input_embeddings()
    embedded = word_embeddings(tokenized.input_ids)
    return embedded
