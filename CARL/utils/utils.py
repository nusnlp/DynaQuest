import json
import random
import torch
import numpy as np

import re
import string
import collections
from datetime import datetime

def read_json(d):
	with open(d, 'r', encoding='utf-8') as f:
		return json.load(f)

def write_json(data, dsave, indent=True):
	outf = open(dsave, 'w', encoding='utf-8')
	if indent:
		json.dump(data, outf, indent=2, ensure_ascii=False)
	else:
		json.dump(data, outf, ensure_ascii=False)
	outf.close()
	print('>>> write to {}'.format(dsave))
	
def write_jsonl(data, dsave):
	outf = open(dsave, 'w')
	for d in data:
		json.dump(d, outf)
		outf.write('\n')
	outf.close()
	print('\n+++ write to {}'.format(dsave))

def load_jsonl_data(data_dir):
	data = []
	with open(data_dir, 'r', encoding='utf-8') as lf:
		for line in lf:
			data.append(json.loads(line))
	lf.close()
	return data

def append_jsonl(d, dsave):
	with open(dsave, 'a') as fout:
		json.dump(d, fout)
		fout.write('\n')

def set_seed(seed, n_gpu=1):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def load_timeqax_data(input_path):
	data = load_jsonl_data(input_path)
	datap = []

	for di, d in enumerate(data):
		question = d['question']
		answer = d['answer']
		paras = d['paragraphs']
		context = ' ||\n'.join([pr['text'] for pr in paras]).strip()
		# context = '\n'.join([pr['text'] for pr in paras]).strip()
		context = ' '.join(context.split(' ')[:7000])
		ans_in_context = True
		if 'ans_in_context' in d.keys():
			ans_in_context = d['ans_in_context']
		dp = {'id': f"timeqax-{di}",
			  'question': question,
			  'input': question,
			  'output': answer,
			  'context': context,
			  'ans_in_context': ans_in_context,
			}
		datap.append(dp)
	
	return datap

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def normalize_time_string(time_string):
    try:
        # Remove trailing periods and extra spaces
        time_string = time_string.strip().rstrip('.')
        
        # Normalize multiple time periods separated by commas, semicolons, or slashes
        if ',' in time_string or ';' in time_string or '/' in time_string:
            separators = [',', ';', '/']
            for sep in separators:
                if sep in time_string:
                    periods = re.split(f'[ {sep}]+', time_string)
                    normalized_periods = [normalize_time_string(period.strip()) for period in periods]
                    return ', '.join(normalized_periods)
        
        # Normalize time periods connected by a dash or "to"
        time_string = time_string.replace("–", "-").replace("—", "-").replace(" -", "-").replace("- ", "-").strip()

        # Handle ranges like "1990-2009" or "1991-present"
        if '-' in time_string:
            parts = time_string.split('-')
            if len(parts) == 2:
                start_date = normalize_time_string(parts[0].strip())
                end_date = parts[1].strip().lower()
                if end_date == "present":
                    return f"{start_date} to present"
                else:
                    end_date = normalize_time_string(end_date)
                    if start_date and end_date:
                        return f"{start_date} to {end_date}"

        # Normalize specific date formats
        date_patterns = [
            (r'^\d{4}, \d{2}, \d{2}$', '%Y, %m, %d'),  # "2001, 08, 24"
            (r'^\d{4}, \d{2}$', '%Y, %m'),  # "2001, 08"
            (r'^\d{1,2} \w{3,9} \d{4}$', '%d %B %Y'),  # "21 October 1929"
            (r'^\w{3,9} \d{1,2}, \d{4}$', '%B %d, %Y'),  # "December 21, 1987"
            (r'^\w{3} \d{1,2}, \d{4}$', '%b %d, %Y'),  # "Mar 13, 2001"
            (r'^\w{3,9} \d{4}$', '%B %Y'),  # "December 1987"
            (r'^\d{4}$', '%Y')  # "1991"
        ]

        for pattern, date_format in date_patterns:
            if re.match(pattern, time_string):
                parsed_time = datetime.strptime(time_string, date_format)
                if date_format == '%Y':
                    return parsed_time.strftime('%Y')
                elif date_format == '%Y, %m':
                    return parsed_time.strftime('%Y-%m')
                elif date_format == '%Y, %m, %d':
                    return parsed_time.strftime('%Y-%m-%d')
                else:
                    return parsed_time.strftime('%Y-%m-%d')
        
        # Default handling for other date formats
        parsed_time = datetime.strptime(time_string, '%B %d, %Y')
        return parsed_time.strftime('%Y-%m-%d')
    
    except ValueError:
        return time_string  # Return None if the input is not a valid date/time

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1