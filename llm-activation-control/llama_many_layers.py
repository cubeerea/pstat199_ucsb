import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc
import numpy as np
import plotly

import random
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List
from transformer_lens import HookedTransformer, HookedTransformerConfig ,utils, ActivationCache
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float, Int
from colorama import Fore
import plotly.graph_objects as go
import plotly.express as px
import string 

import os

# Choose one model for the experimentw
MODEL_PATH = (
    # "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    # "google/gemma-2-9b-it"
    # "google/gemma-2-27b-it"
    # "Unispac/Gemma-2-9B-IT-With-Deeper-Safety-Alignment"
)

MODEL_NAME = MODEL_PATH.split("/")[-1]

# RL: We use GPU 1 from {0, ..., 7}
DEVICE = "cuda:5"

BATCH_SIZE = 512
beta = 0.9
p_coe = 1.0
i_coe = 0.3
d_coe =0.01
noise_prob = 1.0

OUTPUT_PARENT_DIR = Path("output") / f"{MODEL_NAME}" / f"causal_noise"

OUTPUT_DIR = OUTPUT_PARENT_DIR / (f"pid_many_layers_" + str(p_coe).replace(".", "p") + "_" + str(i_coe).replace(".", "p") + "_" + str(d_coe).replace(".", "p") + "_noise_"+str(noise_prob).replace(".", "p")) 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


VISUALIZATION_PARENT_DIR = Path("visualization")

VISUALIZATION_DIR = VISUALIZATION_PARENT_DIR 
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

if MODEL_PATH not in OFFICIAL_MODEL_NAMES:
    OFFICIAL_MODEL_NAMES.append(MODEL_PATH)

CACHE_DIR = Path("/root/.cache/huggingface")
MODEL_CACHE_DIR = CACHE_DIR / "hub"
DATASETS_CACHE_DIR = CACHE_DIR / "datasets"

cfg = HookedTransformerConfig(n_layers = 150,
                              d_model = 768,
                              d_head = 128,
                              n_ctx = 2048,
                              d_mlp = 1532,
                              d_vocab = 128256,
                              act_fn = "silu",
                              normalization_type = "RMS",
                              device = DEVICE,
                              attn_only=False)

model = HookedTransformer(cfg)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.tokenizer = tokenizer

model.tokenizer.padding_side = "left"

# store original chat template
ORIGINAL_CHAT_TEMPLATE = model.tokenizer.chat_template

# handle pad token for some model
if not model.tokenizer.pad_token:
    if "qwen1" in MODEL_PATH.lower():
        model.tokenizer.pad_token = "<|endoftext|>"
    elif model.tokenizer.eos_token:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    else:
        raise ValueError("No pad token found in the tokenizer.")
    
def get_harmful_instructions():
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)

    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = dataset["goal"].tolist()

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions():

    hf_path = "tatsu-lab/alpaca"
    dataset = load_dataset(hf_path, cache_dir = DATASETS_CACHE_DIR)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset["train"])):
        if dataset["train"][i]["input"].strip() == "":
            instructions.append(dataset["train"][i]["instruction"])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmful_instructions_jp():
    huggingface_id = "kunishou/do-not-answer-ja"
    dataset = load_dataset(huggingface_id, cache_dir = DATASETS_CACHE_DIR)

    selected_types = [
        # # Information Hazards
        # "Risks from leaking or inferring sensitive information (organization/gov)",
        # (
        #     "Compromise privacy by leaking or inferring private information"
        #     " (person/individual)"
        # ),
        # Malicious Uses
        "Assisting illegal activities",
        "Nudging or advising users to perform unethical or unsafe actions",
        "Reducing the cost of disinformation campaigns",
        # Discrimination, Exclusion, Toxicity, Hateful, Offensive
        "Social stereotypes and unfair discrimination",
        "Toxic language (hate speech)",
        "Adult Content",
        # # Misinformation Harms
        # "Causing material harm by disseminating misinformation e.g. in medicine or law",
        # "Disseminating false or misleading information",
        # # Human-Chatbot Interaction Harms
        # "Mental Health or Overreliance Crisis",
        # "Treat Chatbot as a Human",
    ]

    instructions = []
    for item in dataset["train"]:
        if item["types_of_harm"] not in selected_types:
            continue
        instructions.append(item["question"])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions_jp():
    huggingface_id = "Lazycuber/alpaca-jp"
    dataset = load_dataset(huggingface_id, cache_dir = DATASETS_CACHE_DIR)

    # filter for instructions that do not have inputs
    instructions = []
    for item in dataset["train"]:
        if item["input"].strip() != "":
            continue
        inst = item["instruction"]
        inst = inst.strip("「」'")
        instructions.append(inst)

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


LANGUAGE = "en"

if LANGUAGE == "en":
    harmful_inst_train, harmful_inst_test = get_harmful_instructions()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions()
elif LANGUAGE == "jp":
    harmful_inst_train, harmful_inst_test = get_harmful_instructions_jp()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions_jp()



# RL: Function Generates the (B, L) inputs to the model w.r.t tokenizer of desired model
def instructions_to_chat_tokens(
    tokenizer: AutoTokenizer,
    instructions: List[str],
) -> Int[Tensor, "batch_size seq_len"]:
    # RL: Checks for if there is a required chat template
    if tokenizer.chat_template:
        # RL: This automatically creates the Batch
        convos = [
            [{"role": "user", "content": instruction}] for instruction in instructions
        ]
        return tokenizer.apply_chat_template(
            convos,
            padding=True, # Padding here ensures all prompts are of the same length
            truncation=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        return tokenizer(
            instructions, padding=True, truncation=False, return_tensors="pt"
        ).input_ids


harmful_sample_toks = instructions_to_chat_tokens(
    tokenizer=model.tokenizer, instructions=harmful_inst_train[:2]
)
harmless_sample_toks = instructions_to_chat_tokens(
    tokenizer=model.tokenizer, instructions=harmless_inst_train[:2]
)

# RL: Important to note how each sample looks like
for sample in harmful_sample_toks[:2]:
    print(model.tokenizer.decode(sample))
    print("-" * 50)
for sample in harmless_sample_toks[:2]:
    print(model.tokenizer.decode(sample))
    print("-" * 50)


def __run_with_cache(model, data, batch_size):
    cache = {}
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            # RL: For this particular function, with respect to input (B, L), it generates the intermediate activations for each block (pre, mid, post), so output will be layer 3*layer keys, each entry is (B, L, D)
            _, batch_cache = model.run_with_cache(
                data[i : i + batch_size],
                names_filter=lambda hook_name: "resid" in hook_name,
                return_cache_object=False,
            )
            for k, v in batch_cache.items():
                if k not in cache:
                    cache[k] = v.cpu()
                else:
                    cache[k] = torch.vstack([cache[k], v.cpu()])
    # RL: Cache Keys are each intermediate location; Each entry is (B, L, D) where B is the total number of prompts
    return ActivationCache(cache, model)


def get_template_suffix_toks(tokenizer):
    # Since the padding is on the left side, the suffix of all samples are the same
    # when using the same template.
    # The activations on these suffix tokens are after the prompt has been processed,
    # thus it's interesting to see how the activations differ between contrastive
    # samples

    # get the common suffix between 2 samples
    toks = instructions_to_chat_tokens(tokenizer=tokenizer, instructions=["a", "b"])

    suffix = toks[0]
    # RL: We traverse backwards
    for i in range(len(toks[0]) - 1, -1, -1):
        # RL: technically it collects the toks from the earliest differing token, but since instruction list is fix to only a and b, it will determinstically obtain the following:
        # RL: "Qwen/Qwen2.5-3B-Instruct" from ids -> token: ['<|im_end|>', 'Ċ', '<|im_start|>', 'assistant', 'Ċ']
        if toks[0][i] != toks[1][i]:
            suffix = toks[0][i + 1 :]

    return tokenizer.convert_ids_to_tokens(suffix)

def add_noise(s, prob=0.3):
    result = []
    for c in s:
        r = random.random()
        if r < prob/3:
            continue  # delete
        elif r < 2*prob/3:
            result.append(random.choice(string.ascii_letters))  # substitute
        else:
            result.append(c)
            if random.random() < prob/2:
                result.append(random.choice(string.ascii_letters))  # insert
    return "".join(result)

def get_activations(
    model: HookedTransformer,
    instructions: List[str],
    batch_size: int = BATCH_SIZE,
    act_names: List[str] = ["resid_mid", "resid_post"], # RL: The Positions we're interested in
    num_last_tokens: int = 1,
):
    # tokenize instructions
    toks = instructions_to_chat_tokens(
        tokenizer=model.tokenizer, instructions=instructions
    )

    
    # run model on instructions and cache activations
    with torch.no_grad():
        cache = __run_with_cache(model, toks, batch_size=BATCH_SIZE)

    # get activations for the last n tokens
    acts = torch.stack(
        [
            torch.stack(
                [cache[act, layer][:, -num_last_tokens:, :] for act in act_names]
            )
            for layer in range(model.cfg.n_layers)
        ]
    )
    # For acts: layers x resid_modules [RL: mid, post] x batch [RL: prompts] x tokens x dim
    return acts, cache


def get_CAA_output_hook(layer, direction: Tensor):
    def hook_fn(output, hook):
        # RL: We want to ablate dest -> src  (Input should be negative of harmful - harmless)
        nonlocal direction
        # nonlocal direction

        # RL: Obtain Activations (Might be a tuple so we obtain the activation component)
        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        # ActAdd alpha_param
        alpha = 1.0
        # RL: Normalize the direction (dir is 1D vector)
        direction = direction / (direction.norm(p = 2) + 1e-8)  
        direction = direction.to(activation)
        activation += alpha * direction

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


import tqdm
N_INST_TRAIN = 512
BATCH_SIZE = 512

# extraction points per decoder block
act_names = ["resid_mid", "resid_post"]

# get the template suffix tokens
template_suffix_toks = get_template_suffix_toks(model.tokenizer)
if not template_suffix_toks:
    template_suffix_toks = ["<last token>"]

# only get the activations of the template suffix tokens since these tokens are the same
# for all samples
# RL: The causal nature implies the output of these tokens already contain the information of the prompts itself.
num_last_tokens = len(template_suffix_toks)
print("template_suffix_toks:", template_suffix_toks)

# RL: File Path Names
chosen_token = -1
refusal_dirs_path = (
    OUTPUT_DIR
    / f"refusal_dirs_{chosen_token}_{LANGUAGE}.npy"
)
unnormed_refusal_dirs_path = (
    OUTPUT_DIR
    / f"refusal_dirs_unnormed_{chosen_token}_{LANGUAGE}.npy"
)

# RL: Harmful should be left alone since we don't steer it and it will be deterministic; Furthermore, this is not unique as the model isn't steered
output_harmful_file = OUTPUT_PARENT_DIR / f"acts_harmful_{LANGUAGE}.npy"

# RL: Harmless should be rerun during learning, and this is unique to each to each choice of beta
output_harmless_file = OUTPUT_DIR / f"acts_harmless_{LANGUAGE}.npy"

import tqdm
N_INST_TRAIN = 512
BATCH_SIZE = 512

# extraction points per decoder block
act_names = ["resid_mid", "resid_post"]

# get the template suffix tokens
template_suffix_toks = get_template_suffix_toks(model.tokenizer)
if not template_suffix_toks:
    template_suffix_toks = ["<last token>"]

# only get the activations of the template suffix tokens since these tokens are the same
# for all samples
# RL: The causal nature implies the output of these tokens already contain the information of the prompts itself.
num_last_tokens = len(template_suffix_toks)
print("template_suffix_toks:", template_suffix_toks)

# RL: File Path Names
chosen_token = -1
refusal_dirs_path = (
    OUTPUT_DIR
    / f"refusal_dirs_{chosen_token}_{LANGUAGE}_{MODEL_PATH.split('/')[-1]}.npy"
)
unnormed_refusal_dirs_path = (
    OUTPUT_DIR
    / f"refusal_dirs_unnormed_{chosen_token}_{LANGUAGE}_{MODEL_PATH.split('/')[-1]}.npy"
)

p_dirs_path = (
    OUTPUT_DIR
    / f"p_dirs_{chosen_token}_{LANGUAGE}_{MODEL_PATH.split('/')[-1]}.npy"
)

i_dirs_path = (
    OUTPUT_DIR
    / f"i_dirs_{chosen_token}_{LANGUAGE}_{MODEL_PATH.split('/')[-1]}.npy"
)

d_dirs_path = (
    OUTPUT_DIR
    / f"d_dirs_{chosen_token}_{LANGUAGE}_{MODEL_PATH.split('/')[-1]}.npy"
)
# RL: Harmful should be left alone since we don't steer it and it will be deterministic; Furthermore, this is not unique as the model isn't steered
output_harmful_file = OUTPUT_DIR / f"acts_harmful_{LANGUAGE}_{MODEL_PATH.split('/')[-1]}.npy"

# RL: Harmless should be rerun during learning, and this is unique to each to each choice of beta
output_harmless_file = OUTPUT_DIR / f"acts_harmless_{LANGUAGE}_{MODEL_PATH.split('/')[-1]}.npy"


noisy_harmful_inst_train = [add_noise(s, prob=noise_prob) for s in harmful_inst_train[:N_INST_TRAIN]]
noisy_harmless_inst_train = [add_noise(s, prob=noise_prob) for s in harmless_inst_train[:N_INST_TRAIN]]

if refusal_dirs_path.exists() and unnormed_refusal_dirs_path.exists() and output_harmless_file.exists():
    print("loading refusal_dirs and unnormed refusal_dirs from file")
    unnormed_refusal_dirs = np.load(unnormed_refusal_dirs_path)
    refusal_dirs = np.load(refusal_dirs_path)
    print("loading harmful and sequentially steered harmless files")
    harmful_acts = np.load(output_harmful_file)
    harmful_acts = torch.from_numpy(harmful_acts)
    harmless_acts = np.load(output_harmless_file)
    harmless_acts = torch.from_numpy(harmless_acts)
else:
    # Momentum_mode
    momentum_mode = True
    v = None
    # breakpoint()
    # RL: Do not apply steering to harmful activation, so running them outside to save computations
    if output_harmful_file.exists():
        harmful_acts = np.load(output_harmful_file)
        harmful_acts = torch.from_numpy(harmful_acts)
    else:
        harmful_acts, cache = get_activations(
            model,
            harmful_inst_train[:N_INST_TRAIN],
            batch_size=BATCH_SIZE,
            act_names=act_names,
            num_last_tokens=num_last_tokens,
        )
        np.save(output_harmful_file, harmful_acts.cpu().float().numpy())
    
    # Normalize and Mean across harmful
    harmful_acts_norm = harmful_acts / harmful_acts.norm(dim=-1, keepdim=True)
    harmful_acts_norm_mean = harmful_acts_norm.mean(dim = 2)

    # Direction will store [layer][mid == 0, post == 1]
    directions = {}

    # Initialize refusal_dirs storage:
    d_model = harmful_acts.shape[-1]
    unnormed_refusal_dirs = torch.zeros(model.cfg.n_layers * len(act_names), d_model)

    # Used Module Name, each format should be (layer_ind, (0/1))
    used_module_names = []
    for l in tqdm.tqdm(range(model.cfg.n_layers * len(act_names))):

        layer, pos = l // 2, l % 2

        # RL: Forward Hook Construction; Separated for layer and position:
        fwd_hooks = []
        for tup in used_module_names:
            ly, mp = tup
            # Note that each direction is from src (harmless) to (harmful);
            # We add, so we take positive direction
            if mp == 0:
                fwd_hooks.append((f"blocks.{ly}.hook_resid_mid", get_CAA_output_hook(ly, directions[ly][mp])))
            elif mp == 1:
                fwd_hooks.append((f"blocks.{ly}.hook_resid_post", get_CAA_output_hook(ly, directions[ly][mp])))
            else:
                raise NotImplementedError("mp not 0 or 1")
        # fwd_hooks = [(f"blocks.{ly}.hook_resid_post", get_direction_ablation_output_hook(ly, directions[ly])) for ly in used_module_names]
            # These hooks will intervene on the model, that's why we need hook_* args.
        # get contranstive activations
        # RL: Apply steering to only Harmless Activation (Src)
        
        with model.hooks(fwd_hooks=fwd_hooks):   
            harmless_acts, cache = get_activations(
                model,
                harmless_inst_train[:N_INST_TRAIN],
                batch_size=BATCH_SIZE,
                act_names=act_names,
                num_last_tokens=num_last_tokens,
            )

    
        # print(harmful_acts.shape)
        # print(harmless_acts.shape)

        # For each step, we find the difference of normed means (Norm the activations, then take the mean) then we find the refusal direction
        # Take the Mean across Batch
        harmless_acts_norm = harmless_acts / harmless_acts.norm(dim=-1, keepdim=True)
        
        # Take Mean across Batch
        harmless_acts_norm_mean = harmless_acts_norm.mean(dim = 2)

        # Take the difference in normed mean (For now, our ref_dir is from harmless -> harmful; but in our directional ablation we take the opposite direction)
        ref_dir_set = harmful_acts_norm_mean - harmless_acts_norm_mean
        # Specifically, choose the last token 
        d_model = ref_dir_set.shape[-1]
        ref_dir_set = ref_dir_set[:, :, -1].reshape(-1, d_model)
        ref_dir = ref_dir_set[l] # This should be = to layer*2 + pos

        shifted_ref_dir = ref_dir_set.roll(1, dims=0)
        shifted_ref_dir[0] = ref_dir_set[0]
        der_comp = ref_dir_set - shifted_ref_dir

        int_comp = torch.cumsum(ref_dir_set, dim=0)
        # seq_harmful_acts_normed = seq_harmful_acts / seq_harmful_acts.norm(dim=-1, keepdim=True)
        # seq_harmless_acts_normed = seq_harmless_acts / seq_harmless_acts.norm(dim=-1, keepdim=True)
        if pos == 0:
            directions[layer] = {}
        if momentum_mode:
            if l == 0:
                v = ref_dir + torch.randn_like(ref_dir) * noise_prob + 0.1
            else:
                
                v = p_coe*ref_dir + i_coe*int_comp[l] + d_coe*der_comp[l] + torch.randn_like(ref_dir) * noise_prob + 0.1
            directions[layer][pos] = v
            unnormed_refusal_dirs[l, :] = v.detach()
        else:
            directions[layer][pos] = ref_dir
            unnormed_refusal_dirs[l, :] = ref_dir.detach()
        # harmful_acts_normed_mean = seq_harmful_acts_normed.mean(dim=2)
        # harmless_acts_normed_mean = seq_harmless_acts_normed.mean(dim=2)
        used_module_names.append((layer, pos))

    # RL: Compute Normed Refusal Directions
    refusal_dirs = unnormed_refusal_dirs / unnormed_refusal_dirs.norm(dim=-1, keepdim=True)
    refusal_dirs = refusal_dirs.reshape(model.cfg.n_layers, len(act_names), d_model).cpu().float().numpy()
    unnormed_refusal_dirs = unnormed_refusal_dirs.reshape(model.cfg.n_layers, len(act_names), d_model).cpu().float().numpy()

    # RL: Set harmless + harmful acts to cpu and float
    harmful_acts = harmful_acts.cpu().float()
    harmless_acts = harmless_acts.cpu().float()

    # Save harmless, refusal directions normed and unnormed
    np.save(output_harmless_file, harmless_acts.numpy())
    np.save(unnormed_refusal_dirs_path, unnormed_refusal_dirs)
    np.save(refusal_dirs_path, refusal_dirs)
    np.save(p_dirs_path, ref_dir_set.cpu().float().numpy())
    np.save(i_dirs_path, int_comp.cpu().float().numpy())
    np.save(d_dirs_path, der_comp.cpu().float().numpy())