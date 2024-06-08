import json
import os
import re
import time
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
ckpt_path = "/home/yerong2/models/internlm-xcomposer2-vl-7b"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')

DEBUG = False
model = model.eval()

# Load the data from the JSON file
FILE='test'

data = {}

# Read the JSONL file
with open(f'data/{FILE}.jsonl', 'r') as file:
    for line in file:
        entry = json.loads(line.strip())  # Parse each line as a JSON object
        data[entry["id"]] = entry
if DEBUG : cnt = 0
for key in tqdm(data):
    entry = data[key]

    # Extract the relevant fields from each entry
    question = data[key]['question']
    cleaned_question = re.sub(r"<image\d+>", "", question).strip()
    re.sub(r"<image\d+>", "", question).strip()
    query_cot = f'Question: {cleaned_question}'
    image_path = entry["image"]
    image = image_path
    
    
    # Define the query for the model
    query = '<ImageHere>' + query_cot

    # Generate the response from the model
    with torch.cuda.amp.autocast():
        response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)

    # Add query CoT to the entry
    entry["query"] = query_cot
    # Add the response to the entry
    entry["model_answer"] = response
    if DEBUG : cnt+= 1
    if DEBUG and cnt > 3: break

# Save the updated data to a new JSON file
file_path = f'data/{FILE}_ans.jsonl'

# Check if the file exists and remove it if it does
if os.path.exists(file_path):
    os.remove(file_path)

# Iterate over the list of values and append each value (dictionary) to the JSON file
with open(file_path, 'w') as file:
    for key in data:
        json.dump(data[key], file, indent=4)
        file.write('\n')  # Ensure each JSON object is on a new line

print(f"Responses saved to {file_path}")

