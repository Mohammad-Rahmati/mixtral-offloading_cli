#!/usr/bin/env python3
"""
Mixtral8x7B-Instruct (https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) CLI Tool
------------------------------

This module, `llm_cli.py`, provides a command-line interface for interacting with Mixtral8x7B-Instruct. 
It allows users to load the model, perform text generation, and access various functionalities of the LLM in a convenient and user-friendly manner.

Features:
- Model Initialization: Load and initialize the LLM with specified configurations.
- Text Generation: Input prompts and receive generated text from the LLM.
- Model Management: Download model weights, update configurations, and handle various model-related tasks.
- Additional Utilities: Additional functionalities like logging, error handling, and custom settings for advanced users.

The CLI is designed to be intuitive, providing clear guidance and instructions for both new and experienced users. 
It leverages Python libraries such as `argparse` for parsing command-line arguments and handles various scenarios gracefully, 
ensuring a smooth user experience.

Usage:
To use this tool, run the script with the required arguments from the command line. For detailed instructions on available commands and options, use the `-h` or `--help` flag.

Example:
    python llm_cli.py --generate "Hello, world!" --config config.json

Note:
This script assumes that all necessary dependencies are installed and Python 3.x is used. For detailed setup instructions and requirements, refer to the README file.

Acknowledgments:
This tool is based on the work found in https://github.com/dvmazur/mixtral-offloading.git. I thank them for their foundational work, which has been instrumental in the development of this CLI tool.

Author: Mo Rahmati
Version: 1.0.0
"""

import argparse
import sys
import subprocess
from subprocess import run, PIPE
import importlib.metadata as metadata
import os
import json

try:
    from packaging import version
except ModuleNotFoundError:
    # If packaging is not installed, install it using pip
    subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging"])
    from packaging import version  # Import again after installation


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
            "\nThe following packages are missing or do not meet the version requirements:"
        )
        for package in missing_packages:
            print(f" - {package}")
        print("\nPlease 'pip install -r requirements.txt' and rerun the script.")
        sys.exit(1)
    else:
        print("All required packages are installed with correct versions.")


def setup_and_save_config(config_filename="config.json"):
    # Check if the configuration file already exists
    if os.path.exists(config_filename):
        with open(config_filename, "r") as file:
            config = json.load(file)
        
        user_choice = input(
            f"Configuration file {config_filename} already exists.\n{json.dumps(config, indent=1)}\n\nDo you want to use it? (yes/no): "
        ).lower()
        if user_choice == "yes":
            print("Using existing configuration.")
            return
        else:
            print("Creating new configuration.")

    # Create new configuration
    config = {
        "model_path": input("Enter the path for model weights (leave blank to download in the source folder): "),

        "other_settings": {
            "setting1": input("Enter value for setting1: "),
            "setting2": input("Enter value for setting2: "),
        },
    }

    # Save the new configuration to a file
    with open(config_filename, "w") as file:
        json.dump(config, file, indent=4)
    print(f"Configuration saved to {config_filename}")


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Mixtral8x7B-Instruct CLI Tool")

    # Add arguments
    parser.add_argument(
        "--opl",
        help="Increasing offload_per_layer (OPL) will decrease GPU VRAM usage, increase RAM usage and decrease generation speed. \
            Decreasing offload_per_layer will have the opposite effect. \
            Assign this to 5 if you have only 12 GB of GPU VRAM",
        type=str,
    )
    parser.add_argument("--config", help="Path to the configuration file", type=str)

    # Parse the arguments
    args = parser.parse_args()


if __name__ == "__main__":
    setup_and_save_config()
    check_requirements()
