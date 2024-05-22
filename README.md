# Introduction

this project implements ReAct with multiple optional LLMs.

# Usage

## install prerequisite packages

```shell
python3 -m pip install -r requirements.txt
```

## run service

```shell
python3 main.py --model (llama3|codellama) [--host 0.0.0.0] [--port 8081]
```

note that the chain of thought may not stop due to understanding limit of a specific LLM.
