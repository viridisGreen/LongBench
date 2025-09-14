import os
from datasets import load_dataset
import torch
import json
from transformers_v4_56_1 import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
from ipdb import set_trace as st
import copy

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=[
        "llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k",
        "llama2-7b", "llama3-8b", "llama3-8b-instruct", "qwen3-8b"
    ])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--cuda_visible_devices', type=str, default=None, help="CUDA visible devices")
    parser.add_argument('--offload_layer_id', type=int, nargs='+', help='List of layer IDs to offload')
    parser.add_argument('--skip_layer_id', type=int, nargs='+', help='List of layer IDs to skip')
    parser.add_argument('--quantize_type', type=str, default='none', choices=['none', 'INT8', 'INT4'], help='Quantization type')
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "llama3" in model_name:
        # Llama 3 Instruct 建议用 apply_chat_template
        # try:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # 让模板自动补 assistant 起始
        )
        # except Exception:
        #     # 兼容老版本 transformers：回退到官方 header 近似格式
        #     # 参考：<|start_header_id|>user ... <|eot_id|> <|start_header_id|>assistant ...
        #     prompt = (
        #         "<|begin_of_text|>"
        #         "<|start_header_id|>user<|end_header_id|>\n"
        #         f"{prompt}"
        #         "<|eot_id|>"
        #         "<|start_header_id|>assistant<|end_header_id|>\n"
        #     )
    elif "qwen3" in model_name:
        # 优先使用仓库内置的 chat_template（通常在 *-Instruct 模型里有）
        tpl = getattr(tokenizer, "chat_template", None)
        if tpl:
            try:
                messages = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,           # 这里只返回字符串；若你想直接返回张量见下方 get_pred 补丁
                    add_generation_prompt=True
                )
                return prompt
            except Exception:
                pass  # 模板不可用时回退到手写

        # ——无模板的安全回退（Qwen 常用 <|im_start|>/<|im_end|>）——
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful, polite and knowledgeable assistant, don't think, just give the answer.<unthink><|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}<unthink><|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    if dist.is_initialized():
        dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama2" in model_name:
        replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    elif "llama3" in model_name:
        # replace_llama_attn_with_flash_attn()
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            use_fast=True,           # 保证走 fast
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16
        ).to(device)
        model = model.eval()
    elif "qwen3" in model_name:
        # Qwen 系列用 Auto* + trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, use_fast=True
        )
        # 兜底 pad_token，避免 generate 警告/报错
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        # transformers>=4.56：用 dtype= 而不是 torch_dtype=
        # 不强行传 attn_implementation，Qwen 在官方实现里无需猴补丁
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, dtype=torch.bfloat16
        ).to(device).eval()

        # 同步模型端 pad_token_id
        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

if __name__ == "__main__":
    # 固定随机种子
    seed_everything(42)

    # 解析参数
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # （可选）打印 CUDA 环境，确认可见设备与映射是否正确
    import os
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.cuda.device_count() =", torch.cuda.device_count())
        print("current logical device = 0,", "name =", torch.cuda.get_device_name(0))

    # 读取配置
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    # 模型与长度
    model_name = args.model
    max_length = model2maxlen[model_name]

    # 选择评测数据集
    if args.e:
        datasets = [
            "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
            "gov_report", "multi_news", "trec", "triviaqa", "samsum",
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
        ]
    else:
        datasets = [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
            "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report",
            "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum",
            "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
            "lcc", "repobench-p"
        ]

    # 输出目录
    os.makedirs("pred", exist_ok=True)
    os.makedirs("pred_e", exist_ok=True)

    # 单进程、顺序评测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in datasets:
        store_path = copy.deepcopy(model_name)
        if args.offload_layer_id:
            ids_str = '-'.join(map(str, args.offload_layer_id))
            store_path += f"_O{ids_str}"
        if args.skip_layer_id:
            ids_str = '-'.join(map(str, args.skip_layer_id))
            store_path += f"_S{ids_str}"
        if args.quantize_type != 'none':
            quant_level = ''.join(filter(str.isdigit, args.quantize_type)) # 从 "INT8" 中提取 "8"
            store_path += f"_Q{quant_level}"
        if args.e:
            # LongBench-E：从 hub 读取
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
            os.makedirs(f"pred_e/{store_path}", exist_ok=True)
            out_path = f"pred_e/{store_path}/{dataset}.jsonl"
        else:
            # v1 本地数据路径（保持你的原始路径）
            file_path = os.path.join("/home/wanghesong/Datasets/LongBench", f"{dataset}.jsonl")
            data = load_dataset('json', data_files={'test': file_path}, split='test')
            if not os.path.exists(f"pred/{store_path}"):
                os.makedirs(f"pred/{store_path}")
            out_path = f"pred/{store_path}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        # 列表化 dataset，避免多次遍历产生的句柄问题
        data_all = [sample for sample in data]

        # 仅主进程运行：rank=0，world_size=1
        get_pred(
            rank=0,
            world_size=1,
            data=data_all,
            max_length=max_length,
            max_gen=max_gen,
            prompt_format=prompt_format,
            dataset=dataset,
            device=device,
            model_name=model_name,
            model2path=model2path,
            out_path=out_path,
        )

        # 数据集之间清理下显存，降低峰值
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

