# Introduction

this project implements ReAct with multiple optional LLMs.

# Usage

## install prerequisite packages

```shell
python3 -m pip install -r requirements.txt
```

## run service

```shell
python3 main.py --model (ada001|babbage001|curie001|davinci001|davinci002|davinci003|gpt3.5|gpt4|mixtral|mistral|solar|llama2|zephyr) [--host 0.0.0.0] [--port 8880]
```

note that the chain of thought may not stop due to understanding limit of a specific LLM.
