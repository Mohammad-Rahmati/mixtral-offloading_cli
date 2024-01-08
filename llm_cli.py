#!/usr/bin/env python3
"""
Mixtral8x7B-Instruct (https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) CLI Tool
------------------------------

This module, `llm_cli.py`, provides a command-line interface for interacting with Mixtral8x7B-Instruct. 
It allows users to load the model, perform text generation, and access various functionalities of the LLM in a convenient and user-friendly manner.

Acknowledgments:
This tool is based on the work found in https://github.com/dvmazur/mixtral-offloading.git. I thank them for their foundational work, which has been instrumental in the development of this CLI tool.

Author: 1110ra
"""
import os

os.system("cls" if os.name == "nt" else "clear")

import warnings

warnings.filterwarnings(
    "ignore", message="Initializing zero-element tensors is a no-op"
)
import argparse
import sys
import subprocess
from subprocess import run, PIPE
import importlib.metadata as metadata
import json
import time

try:
    from packaging import version
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging"])
    from packaging import version

config_filename = "config.json"
repo_id = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"


def check_requirements(requirements_file="requirements.txt"):
    with open(requirements_file, "r") as file:
        requirements = file.readlines()

    missing_packages = []
    for requirement in requirements:
        requirement = requirement.strip()
        if requirement.startswith("#") or not requirement:
            continue

        if requirement.startswith("git+"):
            repo_url = requirement.split("@")[0][4:]
            package_name = repo_url.split("/")[-1].split(".")[0]
            result = run(["pip", "freeze"], stdout=PIPE, text=True)
            if package_name not in result.stdout:
                missing_packages.append(requirement)
        else:
            package_spec = requirement.split("==")
            package_name = package_spec[0]
            if len(package_spec) == 2:
                required_version = package_spec[1]
                try:
                    installed_version = metadata.version(package_name)
                    if version.parse(installed_version) < version.parse(
                        required_version
                    ):
                        missing_packages.append(
                            f"{package_name} (required: {required_version}, found: {installed_version})"
                        )
                except metadata.PackageNotFoundError:
                    missing_packages.append(requirement)

    if missing_packages:
        print(
            "\n\033[34mThe following packages are missing or do not meet the version requirements:\033[0m"
        )
        for package in missing_packages:
            print(f" - {package}")
        print(
            "\n\033[34mPlease 'pip install -r requirements.txt' and rerun the script.\033[0m"
        )
        sys.exit(1)
    else:
        print(
            "\033[34mAll required packages are installed with correct versions.\033[0m"
        )


def setup_and_save_config(config_filename=config_filename):
    # Check if the configuration file already exists
    if os.path.exists(config_filename):
        with open(config_filename, "r") as file:
            config_user = json.load(file)

        user_choice = input(
            f"\033[34mConfiguration file {config_filename} already exists.\033[0m\n\n\033[33m{json.dumps(config_user, indent=1)}\033[0m\n\n\033[32m--> Do you want to use it? (yes/no):\033[0m "
        ).lower()
        if user_choice == "yes":
            print("\033[34mUsing existing configuration.\033[0m")
            return config_user
        else:
            print("\033[34mCreating new configuration.\033[0m")

    # Create new configuration
    config_user = {
        "model_path": input(
            f"\033[32m--> Enter the path for model weights (default: {os.path.join(os.getcwd(), 'model')}):\033[0m "
        ),
        "offload_per_layer": input("\033[32m--> Enter offload per layer:\033[0m ")
    }
    
    if config_user["model_path"] == "":
        config_user["model_path"] = os.path.join(os.getcwd(), "model")

    # Save configuration
    with open(config_filename, "w") as file:
        json.dump(config_user, file, indent=4)
    
    print(f"\033[34mConfiguration saved to {config_filename}\033[0m")
    return config_user

def download_huggingface_model(repo_id=repo_id, model_path=""):
    from huggingface_hub import snapshot_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    try:
        print(f"\033[34mDownloading model weights from {repo_id}\033[0m")
        time.sleep(1)
        file_path = snapshot_download(repo_id=repo_id, cache_dir=model_path)
        for f in os.listdir(model_path):
            if f.startswith("tmp"):
                os.remove(os.path.join(model_path, f))

        print(f"\033[34mModel weights downloaded to {file_path}\033[0m")
        return file_path

    except Exception as e:
        print(f"\033[31mError downloading model weights:\033[0m {e}")


def spinning_wheel():
    anim = [
        "[    ]",
        "[=   ]",
        "[==  ]",
        "[=== ]",
        "[ ===]",
        "[  ==]",
        "[   =]",
        "[    ]",
        "[   =]",
        "[  ==]",
        "[ ===]",
        "[=== ]",
        "[==  ]",
        "[=   ]",
    ]
    while not thread_stop_event.is_set():  # Continue while the stop event is not set
        for frame in anim:
            if thread_stop_event.is_set():
                print("\033[K\r✅", end="\n")
                break
            print(frame, end="\033[K\r")
            sys.stdout.flush()
            time.sleep(0.1)


def main(state_path="", config_user=""):
    import torch
    from torch.nn import functional as F
    from hqq.core.quantize import BaseQuantizeConfig
    from huggingface_hub import snapshot_download
    from tqdm.auto import trange
    from transformers import AutoConfig, AutoTokenizer
    from transformers.utils import logging as hf_logging
    from src.build_model import OffloadConfig, QuantConfig, build_model
    from transformers import TextStreamer
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"

    config = AutoConfig.from_pretrained(quantized_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offload_per_layer = int(config_user["offload_per_layer"])
    num_experts = config.num_local_experts

    offload_config = OffloadConfig(
        main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
        offload_size=config.num_hidden_layers * offload_per_layer,
        buffer_size=4,
        offload_per_layer=offload_per_layer,
    )

    attn_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        quant_zero=True,
        quant_scale=True,
    )
    attn_config["scale_quant_params"]["group_size"] = 256

    ffn_config = BaseQuantizeConfig(
        nbits=2,
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )
    quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)

    model = build_model(
        device=device,
        quant_config=quant_config,
        offload_config=offload_config,
        state_path=state_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    past_key_values = None
    sequence = None
    seq_len = 0
    while True:
        print("User: ", end="")
        user_input = input()
        print("\n")

        user_entry = dict(role="user", content=user_input)
        input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(
            device
        )

        if past_key_values is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
            attention_mask = torch.ones(
                [1, seq_len - 1], dtype=torch.int, device=device
            )

        print("Mixtral: ", end="")
        result = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            streamer=streamer,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        sequence = result["sequences"]
        past_key_values = result["past_key_values"]


if __name__ == "__main__":
    check_requirements()
    config_user = setup_and_save_config()

    if os.path.exists(config_user["model_path"]):
        state_path = os.path.join(
            config_user["model_path"],
            "models--lavawolfiee--Mixtral-8x7B-Instruct-v0.1-offloading-demo/snapshots/3d47c8315811b9e0135d4fac21deb88309c6551c",
        )
    else:
        state_path = download_huggingface_model(
            repo_id=repo_id, model_path=config["model_path"]
        )
    os.system("cls" if os.name == "nt" else "clear")
    main(state_path=state_path, config_user=config_user)
