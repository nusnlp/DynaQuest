import os
import torch
import transformers
import numpy as np
import wandb
from datasets import load_dataset, Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from typing import List
from tqdm import tqdm
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.utils import write_json, append_jsonl, normalize_answer, set_seed, load_timeqax_data
import fire
wandb.init(mode="disabled")

from peft import (
	PeftModel,
	LoraConfig,
	get_peft_model,
	prepare_model_for_kbit_training,
)

LLAMA31_PAD_TOKEN='<|finetune_right_pad_id|>'

TEMPLATE = """
Additional information: {context}
Provide the shortest answer to the question without any introductory phrases or explanations: {question}
"""
TEMPLATE = TEMPLATE.strip()

def train(
	# model/data params
	base_model: str = "",  # the only required argument
	data_path: str = "",
	output_dir: str = "./saved/output",
	# training hyperparams
	batch_size: int = 128,
	micro_batch_size: int = 4,
	num_epochs: int = 3,
	learning_rate: float = 3e-4,
	cutoff_len: int = 256,
	val_set_size: int = 2000,
	# lora hyperparams
	lora_r: int = 8,
	lora_alpha: int = 16,
	lora_dropout: float = 0.05,
	lora_target_modules: List[str] = [
		"q_proj",
		"v_proj",
	],
	train_on_inputs: bool = True,  # if False, masks out inputs in loss
	add_eos_token: bool = False,
	eval_steps=200,
	save_steps=200,
	seed=62,
	debug_mode=False,
	save_total_limit=2,
):
	device_map = "auto"
	world_size = int(os.environ.get("WORLD_SIZE", 1))
	ddp = world_size != 1
	if ddp:
		device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

	source_path = '/path/to/Meta-Llama-3.1-8B-Instruct'

	if '-8B' in base_model:
		model_name = 'llama318inst'

	output_dir = f'saved/lora_{model_name}_policy_r{lora_r}_lr{learning_rate}_b{batch_size}_sd{seed}_general'

	assert not os.path.exists(output_dir)
	os.makedirs(output_dir)
	set_seed(seed=seed)

	load_in_4bit = True
	load_in_8bit = True if not load_in_4bit else False
	print('load_in_4bit:', load_in_4bit)
	print('load_in_8bit:', load_in_8bit)

	bnb_config = transformers.BitsAndBytesConfig(
		load_in_4bit=load_in_4bit,
		load_in_8bit=load_in_8bit,
		bnb_4bit_use_double_quant=True,
		bnb_4bit_quant_type='nf4',
		bnb_4bit_compute_dtype=torch.float16,
	)

	policy_model = AutoModelForCausalLM.from_pretrained(
		source_path,
		torch_dtype=torch.float16,
		quantization_config=bnb_config,
		device_map=device_map,
	)

	peft_config = LoraConfig(
			r=lora_r,
			lora_alpha=lora_alpha,
			target_modules=lora_target_modules,
			lora_dropout=lora_dropout,
			bias="none",
			task_type="CAUSAL_LM",
		)

	ref_model = create_reference_model(policy_model)
	ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_model)

	policy_model = prepare_model_for_kbit_training(policy_model)
	policy_model = get_peft_model(policy_model, peft_config)
	policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(policy_model)

	tokenizer = AutoTokenizer.from_pretrained(source_path)

	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = (
		tokenizer.eos_token_id
	)

	tokenizer.padding_side = "left" 

	def tokenize(prompt, add_eos_token=False, cutoff_len=cutoff_len):
		prompt_x = prompt
		cutoff_len_x = int(cutoff_len//2)
		prompt_words = prompt_x.split(' ')
		n_words = len(prompt_words)
		n_last_words = 70
		if n_words > cutoff_len_x:
			cutoff_len_z = min(int(cutoff_len//2), n_words - n_last_words)
			prompt_words = prompt_words[:cutoff_len_z] + prompt_words[-n_last_words:]
			prompt_x = ' '.join(prompt_words)

		result = tokenizer(
			prompt_x,
			truncation=True,
			max_length=cutoff_len,
			padding=False,
			return_tensors=None,
		)
		# print('len:', len(result["input_ids"]))
		if (
			result["input_ids"][-1] != tokenizer.eos_token_id
			and len(result["input_ids"]) < cutoff_len
			and add_eos_token
		):
			result["input_ids"].append(tokenizer.eos_token_id)
			result["attention_mask"].append(1)

		result["labels"] = result["input_ids"].copy()
		# result["prompt"] = prompt

		return result

	def generate_and_tokenize_prompt(data_point, gen=False, eval=False):
		query = TEMPLATE.format(question=data_point['question'], context=data_point['context'])
		response = data_point['output']
		if eval:
			msg = [{"role": "user", "content": query}, {"role": "assistant", "content": response}]
			formatted_prompt =  tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
			tokenized_full_prompt = tokenize(formatted_prompt, add_eos_token=False, cutoff_len=10000)
		elif gen:
			msg = [{"role": "user", "content": query}]
			formatted_prompt =  tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
			tokenized_full_prompt = tokenize(formatted_prompt, add_eos_token=False, cutoff_len=2000)
		else:
			msg = [{"role": "user", "content": query}, {"role": "assistant", "content": response}]
			formatted_prompt =  tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
			tokenized_full_prompt = tokenize(formatted_prompt, add_eos_token=False, cutoff_len=1000)

		# train_on_inputs: False
		if not train_on_inputs:
			user_msg = [{"role": "user", "content": query}]
			user_prompt = tokenizer.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=False)
			tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
			user_prompt_len = len(tokenized_user_prompt["input_ids"])
			if add_eos_token:
				user_prompt_len -= 1
			tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
		
		tokenized_full_prompt['idx'] = int(data_point['id'].split('-')[-1])
		return tokenized_full_prompt
	
	def generate_and_tokenize_prompt_gen(data_point):
		return generate_and_tokenize_prompt(data_point, gen=True)
	
	def process_query_tensor(qt):
		i = qt.tolist().index(128000)
		return qt[i:]

	def save_check_point():
		pass

	def get_step_logs(stats, batch, rewards, step_i):
		logs = {}
		logs['step'] = step_i
		rewards = torch.stack(rewards)
		for k, v in stats.items():
			if not isinstance(v, np.ndarray):
				logs[k] = v
		logs["env/reward_mean"] = torch.mean(rewards).item()	# torch.mean(rewards).cpu().numpy().item()
		logs["env/reward_std"] = torch.std(rewards).item()		# torch.std(rewards).cpu().numpy().item()
		logs["env/reward_dist"] = rewards.tolist()	# rewards.cpu().numpy()
		return logs
	
	def get_text_logs(pred, ans, epoch, step_i, idx_b):
		logs = []
		for p, a, idx in zip(pred, ans, idx_b):
			d = {'epoch': epoch, 'step': step_i, 'idx': idx, 'output': p}
			logs.append(d)
			idx += 1
		return logs
	
	def process_text_for_reward(tt, first_line=False, process_parenthesis=False):
		if process_parenthesis:
			tt = tt.replace('(', '( ')
			tt = tt.replace(')', ' )')
			tt = tt.replace('(  ', '( ')
			tt = tt.replace('  )', ' )')
		if first_line:
			tt = tt.strip().split('\n')[0]
		tt = tt.strip()
		if tt.endswith('.'):
			tt = tt[:-1].strip()
		return tt

	def get_reward(pred, ans, context, answer_in_context=None):
		if not answer_in_context:
			answer_in_context = [True for _ in range(len(pred))]
		r_scale = 1.0
		rewards = []
		for p, a, ctx, aic in zip(pred, ans, context, answer_in_context):
			# ps = f"{p.strip()}"
			ps = process_text_for_reward(p, first_line=True, process_parenthesis=True)
			aa = process_text_for_reward(a)
			ps_norm = normalize_answer(ps)
			aa_norm = normalize_answer(aa)

			if ps_norm == aa_norm:
				r = 5.0
			elif ps_norm in aa_norm or aa_norm in ps_norm:
				r = 4.0
			elif aic and (ps in ctx):
				r = 2.5
			elif aic and (ps not in ctx):
				r = -2.5
			elif not aic and (ps in ctx):
				r = -2.5
			else:
				r = 1.0

			rewards.append(r*r_scale)

		return rewards

	timeqax_file = 'data/rl_data_train_na.jsonl'

	train_data = load_timeqax_data(timeqax_file)
	train_data = train_data[-30000:]

	if debug_mode:
		train_data = train_data[:100]
		eval_data = eval_data[:100]

	print('convert data ...')
	train_data_meta = copy.deepcopy(train_data)
	train_data_meta = {int(td['id'].split('-')[-1]): td for td in train_data_meta}
	train_data = Dataset.from_list(train_data).map(generate_and_tokenize_prompt_gen)
	train_data = train_data.select_columns(['input_ids', 'attention_mask', 'labels', 'idx'])

	print('Done: convert data ...')

	if not ddp and torch.cuda.device_count() > 1:
		# keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
		ref_model.is_parallelizable = True
		ref_model.model_parallel = True
		policy_model.is_parallelizable = True
		policy_model.model_parallel = True

	_batch_size = micro_batch_size
	gradient_accumulation_steps = 1
	_batch_size = micro_batch_size
	_lr = learning_rate
	config = PPOConfig(
		reward_model=None,
		kl_penalty="kl",
		batch_size=_batch_size,
		mini_batch_size=_batch_size,
		gradient_accumulation_steps=gradient_accumulation_steps,
		ppo_epochs=num_epochs,
		learning_rate=_lr,
		remove_unused_columns=False,
		seed=42,
	)
	
	trainer = PPOTrainer(
		config=config,
		tokenizer=tokenizer,
		model=policy_model,
		ref_model=ref_model,
		dataset=train_data,
		data_collator=transformers.DataCollatorForSeq2Seq(
				tokenizer,
				pad_to_multiple_of=8,
				return_tensors="pt",
				padding=True,
			),
	)

	generation_kwargs = {
		"min_length": -1,
		"top_k": 0.0,
		"top_p": 1.0,
		"do_sample": False,
		"repetition_penalty": 1.0,
		"pad_token_id": tokenizer.eos_token_id,
		"max_new_tokens": 15,
		"eos_token_id": tokenizer.eos_token_id,
	}

	print('show example training data wrapper:')
	print(tokenizer.decode(train_data[0]['input_ids']))

	epochs = 3
	step_i = 0

	for epoch in tqdm(range(epochs), "epoch:", ncols=100):
		for batch in trainer.dataloader:
			query_tensors = batch["input_ids"]
			input_tensors_b = [process_query_tensor(qt) for qt in query_tensors]
			response_tensors_b = []
			for input_tensors in input_tensors_b:
				response_tensors = trainer.generate([input_tensors], return_prompt=False, **generation_kwargs)
				response_tensors_b += response_tensors
			response_b = [tokenizer.decode(rt, skip_special_tokens=True).strip() for rt in response_tensors_b]
			batch['response'] = response_b
			idx_b = batch['idx'].tolist()

			# print(f'step-{step_i} >>>')

			ans_b = [train_data_meta[idx]['output'] for idx in idx_b]
			context_b = [train_data_meta[idx]['context'] for idx in idx_b]
			ans_in_ctx_b = [train_data_meta[idx]['ans_in_context'] for idx in idx_b]
			reward_b = get_reward(pred=response_b,
						 ans=ans_b,
						 context=context_b,
						 answer_in_context=ans_in_ctx_b)
			
			reward_b = [torch.tensor(r, device='cuda:0') for r in reward_b]

			stats = trainer.step([qt for qt in query_tensors], [rt for rt in response_tensors_b], reward_b)

			step_i += 1

			step_logs = get_step_logs(stats, batch, reward_b, step_i)
			append_jsonl(step_logs, os.path.join(output_dir, 'logs.jsonl'))

			text_logs = get_text_logs(pred=response_b, ans=ans_b, epoch=epoch, step_i=step_i, idx_b=idx_b)
			for tlog in text_logs:
				append_jsonl(tlog, os.path.join(output_dir, 'tlogs.jsonl'))

			if step_i % 200 == 0:
				checkpoint_folder_name = f"step-{step_i}"
				checkpoint_dir = os.path.join(output_dir, checkpoint_folder_name)
				os.makedirs(checkpoint_dir)
				trainer.model.save_pretrained(checkpoint_dir, safe_serialization=False)
				step_stats = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats.items()}
				write_json(step_stats, os.path.join(checkpoint_dir, 'reward_stats.json'))

		if step_i > 150000:
			break

if __name__ == "__main__":
	fire.Fire(train)