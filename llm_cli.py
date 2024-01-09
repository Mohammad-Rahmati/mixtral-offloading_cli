#!/usr/bin/env python3
"""
Mixtral8x7B-Instruct CLI Tool
------------------------------
Provides a command-line interface for Mixtral8x7B-Instruct LLM.
"""

import os
import sys
import json
import time
import threading
import subprocess
import logging
import importlib.metadata as metadata
from subprocess import run, PIPE
from packaging import version
import torch
from torch.nn import functional as F
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, TextStreamer
from src.build_model import OffloadConfig, QuantConfig, build_model
from hqq.core.quantize import BaseQuantizeConfig

# Constants
CONFIG_FILENAME = "config.json"
MODEL_PATH = os.path.join(os.getcwd(), "model")
REPO_ID = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
QUANTIZED_MODEL_NAME = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
RELATIVE_STATE_PATH = "models--lavawolfiee--Mixtral-8x7B-Instruct-v0.1-offloading-demo/snapshots/3d47c8315811b9e0135d4fac21deb88309c6551c"


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


def setup_and_save_config():
    # Check if the configuration file already exists
    if os.path.exists(CONFIG_FILENAME):
        with open(CONFIG_FILENAME, "r") as file:
            config_user = json.load(file)

        user_choice = input(
            f"\033[34mConfiguration file {CONFIG_FILENAME} already exists.\033[0m\n\033[33m{json.dumps(config_user, indent=1)}\033[0m\n\033[32m--> Do you want to use it? (yes/no):\033[0m "
        ).lower()
        if user_choice == "yes":
            return config_user
    
    # Create new configuration
    config_user = {
        "offload_per_layer": input("\033[32m--> Enter offload per layer:\033[0m "),
        "temperature": input("\033[32m--> Enter temperature:\033[0m "),
        "top_p": input("\033[32m--> Enter top p:\033[0m "),
        "max_new_tokens": input("\033[32m--> Enter max new tokens:\033[0m "),
    }
    

    # Save configuration
    with open(CONFIG_FILENAME, "w") as file:
        json.dump(config_user, file, indent=4)

    print(f"\033[34mConfiguration saved to {CONFIG_FILENAME}\033[0m")
    return config_user


def download_huggingface_model(repo_id=REPO_ID, config_user=None):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    try:
        print(f"\033[34mDownloading model weights from {REPO_ID}\033[0m")
        time.sleep(1)
        file_path = snapshot_download(
            repo_id=REPO_ID, cache_dir=MODEL_PATH
        )
        for f in os.listdir(MODEL_PATH):
            if f.startswith("tmp"):
                os.remove(os.path.join(MODEL_PATH, f))

        os.system("clear")
        print(f"\033[34mModel weights downloaded to {file_path}\033[0m")
        return file_path

    except Exception as e:
        print(f"\033[31mError downloading model weights:\033[0m {e}")


def spinning_wheel(stop_event, message):
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

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
    while not stop_event.is_set():
        for frame in anim:
            if stop_event.is_set():
                break
            # Print the frame and stay on the same line
            print(f"\033[34m{message} {frame}\033[0m", end="\033[K\r", flush=True)
            time.sleep(0.1)

    final_message = "✅"
    # Clear the line and print the final message
    print(
        "\033[K\r"
        + f"\033[34mWelcome to Mixtral-8x7B-Instruct CLI! {final_message}\033[0m",
        flush=True,
    )

    # Show the cursor again
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def load_model():
    # Load configuration
    config = AutoConfig.from_pretrained(QUANTIZED_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    offload_per_layer = int(config_user["offload_per_layer"])
    num_experts = config.num_local_experts

    # Setup offloading configuration
    offload_config = OffloadConfig(
        main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
        offload_size=config.num_hidden_layers * offload_per_layer,
        buffer_size=4,
        offload_per_layer=offload_per_layer,
    )

    # Attention and FFN quantization configuration
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

    # Build and return the model
    model = build_model(
        device=device,
        quant_config=quant_config,
        offload_config=offload_config,
        state_path=state_path,
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer, device


def generate_tokens(
    model, input_ids, attention_mask, past_key_values, streamer, tokenizer
):
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        streamer=streamer,
        do_sample=True,
        temperature=config_user["temperature"],
        top_p=config_user["top_p"],
        max_new_tokens=config_user["max_new_tokens"],
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_hidden_states=True,
    )


def process_user_input(model, tokenizer, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    past_key_values = None
    sequence = None
    seq_len = 0
    past_key_values = None

    while True:
        print("\033[32mUser:\033[0m ", end="")
        user_input = input()

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

        result = generate_tokens(
            model, input_ids, attention_mask, past_key_values, streamer, tokenizer
        )

        sequence = result["sequences"]
        past_key_values = result["past_key_values"]


if __name__ == "__main__":
    check_requirements()
    config_user = setup_and_save_config()
    os.system("clear")

    if os.path.exists(MODEL_PATH):
        state_path = os.path.join(MODEL_PATH, RELATIVE_STATE_PATH)
    else:
        state_path = download_huggingface_model(
            repo_id=REPO_ID, config_user=config_user
        )

    thread_stop_event = threading.Event()
    wheel_thread = threading.Thread(
        target=spinning_wheel,
        args=(thread_stop_event, "Welcome to Mixtral-8x7B-Instruct CLI! Loading..."),
    )
    wheel_thread.start()
    model, tokenizer, device = load_model()
    thread_stop_event.set()
    wheel_thread.join()

    process_user_input(model, tokenizer, device)
