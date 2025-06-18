import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import time
import torch
import random
from vllm import LLM, SamplingParams
from utils.utils import compute_f1, compute_exact, read_json, load_jsonl_data
from bs4 import BeautifulSoup
import urllib.parse

torch.manual_seed(107)
torch.cuda.manual_seed_all(107)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(107)

template = """
Additional information: {context}
Provide the shortest answer to the question without any introductory phrases or explanations: {question}
"""
template = template.strip()

model_name = './merged_llama318inst'

bsz = 8
tensor_parallel_size = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

if 'merged_llama318inst' in model_name:
	output_prefix = 'LoraL318Inst'

question_format = 'og'

max_tokens = 20
if '_think_' in model_name:
	max_tokens = 384
top_p = 0.9
temperature= 0
frequency_penalty = 0.5
if temperature == 0:
	top_p = 1
	SEED = 0

sampling_params = SamplingParams(temperature=temperature,
								top_p=top_p,
								max_tokens=max_tokens,
								frequency_penalty=frequency_penalty,
								)
print(sampling_params)

enable_chunked_prefill=False
gpu_memory_utilization=0.92
if 'lama-3.1-70B-Instruct' in model_name:
	enable_chunked_prefill=True
	gpu_memory_utilization=0.98

max_model_len_manual=42000

model = LLM(model=model_name,
			gpu_memory_utilization=gpu_memory_utilization,
			tensor_parallel_size=tensor_parallel_size,
			seed=SEED,
			enable_chunked_prefill=enable_chunked_prefill,
			max_model_len=max_model_len_manual,
			)

tokenizer = model.get_tokenizer()

def extract_answer(text, sep='###'):
	spans = text.strip().split(sep)[0].split('new question:')
	template = spans[0]
	question = ''
	if len(spans) > 1:
		question = spans[1]
	template = template.strip()
	question = question.strip()
	return template, question

identifier_list = [
# '2024-08-16_2024-09-01',
# '2024-09-01_2024-09-16',
# '2024-09-16_2024-10-01',
# '2024-10-01_2024-10-16',
# '2024-10-16_2024-11-01',
# '2024-11-01_2024-11-16',
# '2024-11-16_2024-12-01',
]

for identifier in identifier_list:
	data_path = f'/path/to/{identifier}.retrieval.json'

	# context_mode = 'GWfulltextqa'
	context_mode = 'fulltextqa'

	dsave = f'{output_prefix}_{identifier}_googlesearch.jsonl'
	print('to save >>', dsave)

	data = read_json(data_path)

	tdata = []
	for d in data:
		inst = {}
		inst['id'] = d['id']
		inst['page'] = d['page']
		inst['question'] = d['question']
		inst['answer'] = d['answer']
		if inst['answer'] == 'Unknown':
			continue
		inst['type'] = d['type']
		kk = (d['page'], d['question'])

		# d['context'] includes top-5 full text results from Google Search + Wikipedia
		context = '\n'.join([ctx for ctx in d['context']]).strip()
		# context = ' || '.join([ctx for ctx in d['context']]).strip()
		# context = ' ||\n'.join([ctx for ctx in d['context']]).strip()
	
		inst['ref'] =  ' '.join(context.split(' ')[:30000])

		tdata.append(inst)

	print(len(data), len(tdata))

	start_time = time.time()
	n_count = 0
	matched_count = 0
	start_idx = 0

	tdata = tdata[start_idx:]

	batch_data = [tdata[i:i+bsz] for i in range(0, len(tdata), bsz)]

	current_samples = {}

	sum_em = 0
	sum_f1 = 0

	for bd in batch_data:
		inp_b = []
		for td in bd:
			question = td['question']
			query = template.format(context=td['ref'], question=question)
			if output_prefix.endswith('Inst'):
				msg = [{"role": "user", "content": query}]
				formatted_prompt =  tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
			else:
				formatted_prompt = query + '\nAnswer:'
			inp_b.append(formatted_prompt)

		response_b = model.generate(prompts=inp_b, sampling_params=sampling_params)

		for td, response in zip(bd, response_b):
			raw_output = response.outputs[0].text
			if not output_prefix.endswith('Inst'):
				raw_output = raw_output.rsplit('\n', 1)[0].strip()
			if raw_output.strip().endswith(' |'):
				raw_output = raw_output.split('|')[0].strip()

			inst = {}
			inst['id'] = td['id']
			inst['page'] = td['page']
			target_question = td['question']
			print(td['id'], td['question'])
			inst['question'] = td['question']
			inst['answer'] = td['answer']
			inst['output'] = raw_output

			# em and f1 here are pre-calibrated and not accurate
			inst['em'] = compute_exact(td['answer'], inst['output'])
			inst['f1'] = compute_f1(td['answer'], inst['output'])
			inst['type'] = td['type']
			print('>>> ans:', td['answer'])
			print('>>> raw:', raw_output)
			# print('>>> EM: {:.4f} F1: {:.4f}'.format(inst['em'], inst['f1']))

			fout = open(dsave, 'a')
			json.dump(inst, fout)
			fout.write('\n')
			fout.close()

			n_count += 1
			sum_em += inst['em']
			sum_f1 += inst['f1']

			time_total = time.time() - start_time
			print('====== {:.2f} sec / {} = {:.2f} sec per sample'.format(time_total, n_count, time_total/n_count))
			print('====== EM={:.4f} / F1={:.4f}'.format(sum_em/n_count, sum_f1/n_count))

	print('>>>', dsave)