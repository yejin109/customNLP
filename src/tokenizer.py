import torch
import operator
import functools
from tqdm import tqdm

from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
from kiwipiepy.sw_tokenizer import SwTokenizer, SwTokenizerConfig


def _update_user_word(kiwi, userWords, logger=None):
    for new_word in tqdm(userWords, desc='Add User word'):
        # TODO: support specific score 
        result = kiwi.add_user_word(new_word, 'NNP', 0)

        # TODO: 
        if logger is not None:
            logger.info(f'Add {new_word}, is_sucess {result}')
    return kiwi


def get_from_pretrained(tk_path: str, userWords=None, num_workers=None, logger=None):
    if userWords is None:
        kiwi = None
    else:
        kiwi = Kiwi()
        kiwi = _update_user_word(kiwi, userWords, logger)

        
    tokenizer = SwTokenizer(tk_path, kiwi=kiwi, num_workers=num_workers)
    return tokenizer


def get_from_config(texts, tk_config, save_path, train_cfg, userWords=None, logger=None):
    if userWords is None:
        kiwi = None
    else:
        kiwi = Kiwi()
        kiwi = _update_user_word(kiwi, userWords, logger)

    tokenizer = tokenizer.train(
        save_path=save_path,
        texts =texts,
        config=tk_config,
        kiwi=kiwi,
        **train_cfg # Voca size
        )
    

# TODO: add logger process
# NOTE: functions to be used in Huggingface framework
def tokenize(examples, tokenizer):
    # prompt = tokenizer.encode(examples['prompt'])
    # print(prompt)
    prompt = list(map(lambda batch : tokenizer.encode(batch), examples['prompt']))
    return {'prompt_ids' : prompt}


def batching(examples, ctx_len):
    # print(examples)
    input_batch = []

    all_input_ids = functools.reduce(operator.iconcat, examples["prompt_ids"], [])

    total_length = max((len(all_input_ids) // ctx_len) * ctx_len, 1)

    result = {
        'input_ids' : list(map(lambda i: all_input_ids[i: i+ctx_len], range(0, total_length, ctx_len))),
        # TODO
        # 'attention_mask' : list(map(lambda i: all_attn_masks[i: i+ctx_len], range(0, total_length, ctx_len))),
    }

    return result


def masking(examples, mask_prob, tokenizer):
    inputs = examples['input_ids']
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mask_prob)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).to(inputs.device)
    inputs[indices_random] = random_words[indices_random]

    return {'input_ids': inputs, 'labels': labels}