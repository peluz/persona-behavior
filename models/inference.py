from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM
import torch
import openai
import random
import time

def load_model(model_url):
    # if num_gpus > 1:
    #     with init_empty_weights():
    #         config = AutoConfig.from_pretrained(model_url)
    #         model = AutoModelForCausalLM.from_config(config)
    #     weights_location = snapshot_download(repo_id=model_url, allow_patterns=["pytorch_model*"])
    #     model = load_checkpoint_and_dispatch(
    #         model, weights_location, device_map="auto", dtype=torch.bfloat16, no_split_module_classes=["MistralDecoderLayer"]
    #     )
    # else:
    model = AutoModelForCausalLM.from_pretrained(model_url, torch_dtype=torch.bfloat16, device_map="auto")
    return model

def get_preds(prompts, tokenizer, model, data_collator,  bs=1, max_new_tokens=256, do_sample=False, temperature=1.0, top_p=1.0):
    dataset = Dataset.from_list([{"prompt": prompt} for prompt in prompts])
    tokenized_prompts = dataset.map(lambda x: tokenizer(x["prompt"], truncation=True),
                                    remove_columns=dataset.column_names,
                                    batched=True)
    loader = DataLoader(tokenized_prompts, batch_size=bs, shuffle=False, collate_fn=data_collator)
    all_preds = []
    for batch in tqdm(loader):
        outs = model.generate(input_ids=batch["input_ids"].to(0), attention_mask=batch["attention_mask"].to(0),
                temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample, top_p=top_p)
        all_preds.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in outs])
    return all_preds

def openai_request(client, model_url, prompts, seed=42, temperature=0.0, top_p=1.0, max_tokens=256):
    responses = []
    for request in tqdm(prompts):
        while True:
            num_retries = 0
            delay = 1.
            try:
                response = client.chat.completions.create(
                model=model_url,
                messages=request,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                top_p=top_p
                )
                responses.append(response)
                break
            except Exception as e:
                num_retries += 1
                print(e)
 
                # Check if max retries has been reached
                if num_retries > 10:
                    raise Exception(
                        f"Maximum number of retries (10) exceeded."
                    )
 
                # Increment the delay
                delay *= 2 * (1 + random.random())
                print(f"Retrying with delay {delay}")
 
                # Sleep for the delay
                time.sleep(delay)
    all_preds =  [[{"generated_text": response.choices[0].message.content}] for response in responses]
    responses_json = [response.model_dump_json() for response in responses]
    return responses_json, all_preds