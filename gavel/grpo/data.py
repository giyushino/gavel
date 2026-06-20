#conda_env: verl
import os
import json
from datasets import Dataset, load_dataset, Dataset

import argparse
import re

CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_PATH)

def chat_template(question):
    prompt = "<|im_start|>system\nPlease reason step by step, and present the answer in LaTeX format: \\boxed{PersonA is a knight/knave, PersonB is a knight/knave, ...} listing every person and their role.<|im_end|>\n"
    prompt += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def load_knights_knave(mode: str, turn_off_thinking=False):
    if mode == "train":
        dataset = load_dataset('K-and-K/knights-and-knaves', 'train')
    elif mode == "eval":
        dataset = load_dataset('K-and-K/knights-and-knaves', 'test')

    reformatted = []
    for i in range(2, 5):
        for element in dataset[f"{i}ppl"]:
            prompt = element["quiz"]
            pairs = re.findall(r'(\w+) is a (knight|knave)', element["solution_text"])
            print(pairs)
            result = dict(pairs)
            new_data = {"prompt": chat_template(prompt), "ground_truth": json.dumps(result)}
            reformatted.append(new_data)

    return Dataset.from_list(reformatted)


if __name__ == '__main__':
    # example code for turing gsm8k into parquet file for verl training

    if not os.path.isdir(os.path.join(PROJECT_ROOT, "datasets")):
        os.makedirs(os.path.join(PROJECT_ROOT, "datasets"))

    dataset_name = f"knights_knave_easier"
    dataset_save_path = os.path.join(PROJECT_ROOT, "datasets", dataset_name)

    train_dataset = load_knights_knave("train")
    test_dataset = load_knights_knave("eval")

    print(train_dataset[0])

    # Construct a `def make_map_fn(split)` for the corresponding datasets.
    def make_map_fn(split):
        def process_fn(example, idx):
            #question = chat_template(example["problem"])
            #answer = example["answer"]
            question = example["prompt"]
            # make this a string or else when we grade it won't work

            answer = str(example["ground_truth"])
            data = {
                "data_source": f"{dataset_name}",
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(dataset_save_path, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(dataset_save_path, 'test.parquet'))
    print(f"datasets saved to {dataset_save_path}")

