from transformers import AutoModelForMaskedLM, AutoConfig


def get_model(name_or_path, tokenizer):
    config = AutoConfig.from_pretrained(
        name_or_path,
        vocab_size=len(tokenizer),
        # n_ctx=ctx_len,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = AutoModelForMaskedLM.from_config(config)
    return model