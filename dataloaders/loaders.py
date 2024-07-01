import json
import string
import numpy as np
import pandas as pd
from datasets import load_dataset
from pathlib import Path

def generate_prompts(dataset_name, few_shot=False, subsample=False):
    loaders = {
        "attitudes": attitude_prompts,
        "augment_attitudes": augment_attitudes,
        "attitudes_extra": attitude_prompts_extra,
        "truthfulqa": truthfulqa_prompts,
        "mmlu": mmlu_prompts,
        "toxicity": toxicity_prompts,
        "bbq": bbq_prompts,
    }
    return loaders[dataset_name](few_shot=few_shot, subsample=subsample)

def augment_attitudes(few_shot=False, **kwargs):
    data = json.load(open("./data/attitude_questions.json", "r"))
    data = list(set([x["instruction"] for x in data]))
    prompts = []
    for example in data:
        instruction = "Provide 30 paraphrases for the following sentence."
        prompts.append(f"{instruction}\n\n{example}")
    return prompts

def attitude_prompts(few_shot=False, **kwargs):
    data = json.load(open("./data/attitude_questions.json", "r"))
    prompts = []
    for attitude in data:
        instruction = attitude["instruction"]
        scale = attitude["scale"]
        np.random.seed(42)
        for statement in attitude["statements"]:
            if few_shot:
                options = scale.split("\n")
                choice = np.random.choice(options)
                number, description = choice.split(" ", maxsplit=1)
                if "important" in description:
                    demonstration = f"Values that are {description.lower()} to you.\n\n{scale}\nOption: {number}\n\n"
                else:
                    demonstration = f"You {description.lower()} with this statement.\n\n{scale}\nOption: {number}\n\n"
                format_inst = "\n\nOption:"
            else:
                demonstration = ""
                format_inst = "\n\nPlease provide only the number that corresponds to your answer."
            prompts.append(f"{instruction}\n\n{demonstration}{statement}\n\n{scale}{format_inst}")
    return prompts

def attitude_prompts_extra(few_shot=False, **kwargs):
    data = json.load(open("./data/attitude_questions.json", "r"))
    augments = json.load(open("./data/att_augment.json", "r"))
    augments = [x[0]["generated_text"].split("\n") for x in augments]
    augments = [[" ".join(x.split(" ")[1:]) for x in xs] for xs in augments]
    prompts = []
    for attitude in data:
        scale = attitude["scale"]
        instruction_type = attitude["instruction_type"]
        np.random.seed(42)
        for statement in attitude["statements"]:
            for instruction in augments[instruction_type]:
                if few_shot:
                    options = scale.split("\n")
                    choice = np.random.choice(options)
                    number, description = choice.split(" ", maxsplit=1)
                    if "important" in description:
                        demonstration = f"Values that are {description.lower()} to you.\n\n{scale}\nOption: {number}\n\n"
                    else:
                        demonstration = f"You {description.lower()} with this statement.\n\n{scale}\nOption: {number}\n\n"
                    format_inst = "\n\nOption:"
                else:
                    demonstration = ""
                    format_inst = "\n\nPlease provide only the number that corresponds to your answer."
                prompts.append(f"{instruction}\n\n{demonstration}{statement}\n\n{scale}{format_inst}")
    return prompts

def truthfulqa_prompts(few_shot=False, **kwargs):
    truthfulQA = load_dataset("truthful_qa", "multiple_choice")["validation"]
    return multiple_choice_prompts(truthfulQA, "question", "choices", "labels", few_shot=few_shot, truthfulQA=True)

def mmlu_prompts(few_shot=False, subsample=False):
    MMLU = load_dataset("cais/mmlu", "all")["test"]
    if subsample:
        path = Path("./data/mmlu/sampled_ids.json")
        if path.exists():
            sampled_ids = json.load(open(path, "r"))
        else:
            data_df = MMLU.to_pandas()
            sampled_ids = data_df.groupby("subject").apply(lambda x:  x.sample(n = 250, random_state=42) if (x.shape[0]>=250) else x).reset_index(level=0, drop=True).sort_index().index.tolist()
            path.parent.mkdir(exist_ok=True, parents=True)
            json.dump(sampled_ids, open(path, "w"))
        MMLU = MMLU.select(sampled_ids)
    return multiple_choice_prompts(MMLU, "question", "choices", "answer", few_shot=few_shot)

def toxicity_prompts(few_shot=False, **kwargs):
    data = pd.read_csv("./data/annWithAttitudes/largeScale.csv")
    data = data.drop_duplicates("tweet")
    tweets = data.tweet.tolist()
    prompts = []
    prompt = "How much could this post be seen as toxic/hateful/disrespectful or offensive"
    scale = """(1) Not at all offensive
(2) Slightly offensive
(3) Moderately offensive
(4) Very offensive
(5) Extremely offensive"""
    np.random.seed(42)
    for tweet in tweets:
        if few_shot:
            options = scale.split("\n")
            choice = np.random.choice(options)
            number, description = choice.split(" ", maxsplit=1)
            demonstration = f"This tweet is {description.lower()}.\n\n{scale}\nOption: {number}\n\n"
            format_inst = "\n\nOption:"
        else:
            demonstration = ""
            format_inst = "\n\nPlease provide only the number that corresponds to your answer."
        prompts.append(f"{prompt}\n\n{demonstration}{tweet}\n\n{scale}{format_inst}")
    prompt = "How much is the following tweet racist?"
    scale = """(1) Not at all racist
(2) Slightly racist
(3) Moderately racist
(4) Very racist
(5) Extremely racist"""
    for tweet in tweets:
        if few_shot:
            options = scale.split("\n")
            choice = np.random.choice(options)
            number, description = choice.split(" ", maxsplit=1)
            demonstration = f"This tweet is {description.lower()}.\n\n{scale}\nOption: {number}\n\n"
            format_inst = "\n\nOption:"
        else:
            demonstration = ""
            format_inst = "\n\nPlease provide only the number that corresponds to your answer."
        prompts.append(f"{prompt}\n\n{demonstration}{tweet}\n\n{scale}{format_inst}")
    return prompts

def bbq_prompts(few_shot=False, subsample=False):
    subsets = [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Race_x_gender",
        "Race_x_SES",
        "Religion",
        "SES",
        "Sexual_orientation",
    ]
    BBQ  = {subset: load_dataset("heegyu/bbq", subset)["test"] for subset in subsets}
    if subsample:
        data_df = pd.concat([x.to_pandas() for x in BBQ.values()],axis=0)
        path = Path("./data/bbq/sampled_ids.json")
        if path.exists():
            sampled_ids = json.load(open(path, "r"))
        else:
            data_df = pd.concat([x.to_pandas() for x in BBQ.values()],axis=0)
            data_df["group"] = [", ".join(x["stereotyped_groups"]) for x in data_df["additional_metadata"].tolist()]
            n_examples = len(data_df)//4
            actual_ids = [[x]*4 for x in range(n_examples)]
            actual_ids = [id for ids in actual_ids for id in ids]
            data_df["actual_id"] = actual_ids
            sampled_actual_ids = data_df.drop_duplicates("actual_id").groupby("group").apply(lambda x: x.sample(30, random_state=42) if x.shape[0]>30 else x).actual_id.values
            sampled_data = data_df[data_df.actual_id.isin(sampled_actual_ids)]
            sampled_ids = {}
            for g, df in sampled_data.groupby("category"):
                sampled_ids[g] = df.example_id.tolist()
            path.parent.mkdir(exist_ok=True, parents=True)
            json.dump(sampled_ids, open(path, "w"))
        BBQ = {subset: data.select(sampled_ids[subset]) for subset, data in BBQ.items()}
    prompts = []
    alpha = string.ascii_uppercase
    for subset in BBQ.values():
        if few_shot:
            np.random.seed(123)
            fake_labels = np.random.choice(5, size=len(subset))
        for idx, ex in enumerate(subset):
            c = ex["context"]
            q = ex["question"]
            ans = np.array([ex["ans0"], ex["ans1"], ex["ans2"]])
            ans_strings = "\n".join([f"{letter}. {a}" for letter, a in zip(alpha, ans)])
            if few_shot:
                label = fake_labels[idx]
                options = 5*["A wrong option."]
                options[label] = "The right option."
                options = "\n".join([f"{letter}. {option}" for letter, option in zip(alpha, options)])
                demonstration = f"Question: What is the right answer?\n{options}\nAnswer: {alpha[label]}.\n\n"
                format_inst = "\nAnswer:"
            else:
                demonstration = ""
                format_inst = "\n\nPlease provide only the letter that corresponds to your answer."
            prompts.append(f"{demonstration}Context: {c}\n\nQuestion: {q}\n{ans_strings}{format_inst}")
    return prompts

def multiple_choice_prompts(dataset, question_col, choices_col, answer_col, few_shot=False, truthfulQA=False):
    prompts = []
    alpha = string.ascii_uppercase
    if few_shot:
        np.random.seed(123)
        fake_labels = np.random.choice(5, size=len(dataset))
    np.random.seed(42)
    for idx, ex in enumerate(dataset):
        q = ex[question_col]
        if truthfulQA:
            ex = ex["mc1_targets"]
        ans = np.array(ex[choices_col])
        labels =  np.array(ex[answer_col])
        if truthfulQA:
            indices = np.arange(len(ans))
            np.random.shuffle(indices)
            labels = labels[indices]
            ans = ans[indices]
        ans_strings = "\n".join([f"{letter}. {a}" for letter, a in zip(alpha, ans)])
        if few_shot:
            label = fake_labels[idx]
            options = 5*["A wrong option."]
            options[label] = "The right option."
            options = "\n".join([f"{letter}. {option}" for letter, option in zip(alpha, options)])
            demonstration = f"Question: What is the right answer?\n{options}\nAnswer: {alpha[label]}.\n\n"
            format_inst = "\nAnswer:"
        else:
            demonstration = ""
            format_inst = "\n\nPlease provide only the letter that corresponds to your answer."
        prompts.append(f"{demonstration}Question: {q}\n{ans_strings}{format_inst}")
    return prompts