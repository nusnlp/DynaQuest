import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

import re
import subprocess

def extract_date_range(filename):
    pattern = r"(\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2})"
    match = re.search(pattern, filename)
    if match:
        return match.group(0)
    else:
        return None

lora_path = '/path/to/saved/lora_llama318inst_policy_r64_lr2e-07_b2_sd62_general/step-'

source_path = '/path/to/Meta-Llama-3.1-8B-Instruct'

dest_path = './merged_llama318inst'

identifier = extract_date_range(lora_path)
if identifier:
    dest_path += f'_{identifier}'

dest_path_folder = os.path.dirname(dest_path)
if not os.path.exists(dest_path_folder):
    os.makedirs(dest_path_folder)

subprocess.run(f"cp {source_path}/*tokenizer* {dest_path}/", shell=True)
print('>>> dest_path >>>', dest_path)

base_model = AutoModelForCausalLM.from_pretrained(
    source_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
    trust_remote_code=True,
)

lora_model = PeftModel.from_pretrained(
    base_model,
    lora_path,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

lora_model = lora_model.merge_and_unload()

lora_model.save_pretrained(dest_path, safe_serialization=True, max_shard_size="5GB")
