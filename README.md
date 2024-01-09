# `llm_cli.py` Module README

## Introduction

The `llm_cli.py` module is a command-line interface designed for seamless interaction with the Mixtral8x7B-Instruct Large Language Model (LLM). It provides a straightforward, efficient way for users to engage with the model, facilitating text generation and other functionalities.

## Features

- **Model Loading:** Effortlessly load the Mixtral8x7B-Instruct model.
- **Text Generation:** Utilize the model for generating text based on user inputs.
- **Access to LLM Functionalities:** Explore various capabilities of the LLM in a user-friendly format.

## Acknowledgments

Special thanks to the team at [dvmazur/mixtral-offloading](https://github.com/dvmazur/mixtral-offloading.git) for their foundational work. Their contributions have been crucial in the development of this CLI tool.

## Installation Guide

### Step 1: Create a New Environment

Before installation, it is recommended to set up a new virtual environment to avoid any conflicts with existing packages.

### Step 2: pip install -r requirements.txt

### Step 3: Install `hqq_aten`

Navigate to the `src` directory and run the following commands:

```bash
python3 setup_hqq_aten.py bdist_wheel
pip install dist/hqq_aten-<version>-<tags>.whl
```

## Run
python llm_cli.py