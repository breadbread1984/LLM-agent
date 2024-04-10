#!/usr/bin/python3

from huggingface_hub import login, snapshot_download

login(token = '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')
for model in {'meta-llama/Llama-2-70b-chat-hf',
              'mistralai/Mixtral-8x7B-Instruct-v0.1',
              'OpenHermes-2.5-Mistral-7B',
              'HuggingFaceH4/zephyr-7b-beta',
              'SOLAR-10.7B-Instruct-v1.0'}:
  snapshot_download(repo_id = model)

