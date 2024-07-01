import argparse
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.pipelines.pt_utils import KeyDataset
from dataloaders.loaders import generate_prompts
from pathlib import Path
from datasets import Dataset
from utils.util import initialize_seeds
from tqdm.auto import tqdm
from models.inference import openai_request
from openai import OpenAI
import json
import config
import gc

def main(model_url, openai, few_shot, personas_path, dataset, bs, max_new_tokens, do_sample, temperature, top_p, control, subsample):
    # tokenizer = AutoTokenizer.from_pretrained(model_url, padding_side="left")
    num_gpus = torch.cuda.device_count()
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    print(num_gpus)
    # model = load_model(model_url, num_gpus)
    prefix = "/control" if control else ""
    if not openai: 
        if "mixtral" in model_url.lower():
            quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )   
            model = AutoModelForCausalLM.from_pretrained(model_url, attn_implementation="flash_attention_2", device_map="auto", quantization_config=quant_config)
            tokenizer = AutoTokenizer.from_pretrained(model_url)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
        elif "zephyr" in model_url.lower() or "mistral" in model_url.lower() or "gemma" in model_url.lower():
            model = AutoModelForCausalLM.from_pretrained(model_url, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_url)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
        else:
            pipe = pipeline("text-generation", model=model_url, device_map="auto")
        try:
            has_system = pipe.tokenizer.use_default_system_prompt
        except AttributeError:
            has_system = False
        if pipe.tokenizer.pad_token is None: pipe.tokenizer.pad_token_id = model.config.eos_token_id
    else:
        client = OpenAI(organization=config.org, api_key=config.open_ai_token)
        has_system = True
    if personas_path is None:
        personas_path = "./data/personas.json" if not control else "./data/control_personas.json"
    personas = json.load(open(personas_path, "r"))["personas"]
    datasets = ["attitudes", "truthfulqa", "mmlu", "toxicity", "bbq"] if dataset == "all" else [dataset]
    folder = "zero" if not few_shot else "few"
    
    for dataset_name in datasets:
        for persona in tqdm(personas):
            result_folder = Path(f"./results{prefix}")/folder/model_url.split("/")[-1]/dataset_name
            result_path = result_folder/f'{persona.replace(" ", "_")}.json' if persona != "" else result_folder/"empty.json"
            if result_path.exists(): continue
            initialize_seeds()
            print(f"Generating responses using model {model_url} for dataset {dataset_name} and persona {persona}\n\n")
            prompts = generate_prompts(dataset_name, few_shot=few_shot, subsample=subsample)
            persona_prompt = f"You are {persona}." if len(persona) > 0 else ""
            inputs = []
            if has_system:
                for prompt in prompts:
                    input = [
                                {
                                    "role": "system",
                                    "content": f"{persona_prompt}",
                                },
                                {"role": "user", "content": prompt},
                            ]
                    inputs.append(input)
            else:
                for prompt in prompts:
                    input = [
                        {
                            "role": "user",
                            "content": f"{persona_prompt}\n\n{prompt}",
                        },
                    ]
                    inputs.append(input)
            if not openai:
                inputs = [pipe.tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in inputs]
                # preds = get_preds(inputs, tokenizer=tokenizer, model=model, data_collator=data_collator, bs=bs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p)
                dataset = Dataset.from_list([{"prompt": prompt} for prompt in inputs])
                while True:
                    try:
                        preds = []
                        for out in tqdm(pipe(KeyDataset(dataset, "prompt"), max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p, return_full_text=False, batch_size=bs), total=len(dataset)):
                            preds.append(out)
                        break
                    except torch.cuda.OutOfMemoryError:
                        gc.collect()
                        torch.cuda.empty_cache()
                        bs //= 2
                    
            else:
                if not do_sample: temperature = 0.
                responses, preds = openai_request(client, model_url, inputs, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
                response_path = Path(result_path.as_posix().replace("results", "responses"))
                response_path.parent.mkdir(exist_ok=True, parents=True)
                with open(response_path, "w") as file:
                    json.dump(responses, file)
            gc.collect()
            torch.cuda.empty_cache()
            result_path.parent.mkdir(exist_ok=True, parents=True)
            with open(result_path, "w") as file:
                    json.dump(preds, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get the predictions for a given model and batch size.')
    parser.add_argument('model_url', help='The model to be prompted.', type=str)
    parser.add_argument("--openai", help="Prompt models using openai API",type=str, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--few_shot", help="Add demonstration to prompt.",type=str, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--personas_path', help='A path to file with the personas to be induced.', type=str, default=None)
    parser.add_argument('--dataset', help='The dataset to be processed.', type=str, choices=config.datasets_list + ["all"], default="attitudes")
    parser.add_argument('--bs', help='Prompts per batch', type=int, default=1)
    parser.add_argument('--do_sample', help='Sample tokens when generating.', type=str,
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--max_new_tokens', help='Max number of tokens to generate.', type=int, default=100)
    parser.add_argument('--temperature', help='Temperature for probabiliy scaling.', type=float, default=1.0)
    parser.add_argument('--top_p', help='Top-p proability of tokens for nucleus sampling', type=float, default=1.0)
    parser.add_argument("--control", help="Using control personas",type=str, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subsample", help="Subsample mmlu and bbq",type=str, action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    main(args.model_url, args.openai, args.few_shot, args.personas_path, args.dataset, args.bs, args.max_new_tokens, args.do_sample, args.temperature, args.top_p, args.control, args.subsample)