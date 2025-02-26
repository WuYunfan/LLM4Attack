import os
import numpy as np
from mpmath import nprint

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import transformers
import torch


def encode_feats_2_vectors(path):
    model_id = 'Qwen/Qwen2.5-7B-Instruct'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', device_map='auto')
    feats_tensor = []
    with open(os.path.join(path, 'feats.txt'), 'r') as f:
        feat = f.readline().strip()
        while feat:
            p = feat.find(' ')
            feat = feat[p + 1:]
            messages = [
                {'role': 'system',
                 'content': "As an intelligent book recommender system, your task is to generate a compelling, well-structured, and informative book recommendation summary. "
                            "You should not only present the key information provided but also supplement it with relevant insights based on your own knowledge. "
                            "Ensure that the summary is engaging, concise, and appeals to the target audience. "
                            f"Below is the book's key information: \n{feat}\n"}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                '''
                generated_ids = model.generate(**model_inputs, max_length=np.inf, do_sample=False)
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(response)
                '''
                feat = model(**model_inputs, output_hidden_states=True)
                feat = feat.hidden_states[-1][0, -1, :].cpu()
                feats_tensor.append(feat)
            feat = f.readline().strip()
    feats_tensor = torch.stack(feats_tensor, dim=0)
    torch.save(feats_tensor, os.path.join(path, 'feats.pt'))
    print(feats_tensor[0, :], feats_tensor.dtype)
    feats_tensor = torch.load(os.path.join(path, 'feats.pt'))
    print(feats_tensor[0, :], feats_tensor.dtype)


def main():
    encode_feats_2_vectors('data/Amazon/time')


if __name__ == '__main__':
    main()