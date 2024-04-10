#!/usr/bin/python3

from langchain.llms import HuggingFaceEndpoint, OpenAI
from langchain.agents import load_tools, initialize_agent

class Agent(object):
  def __init__(self, model_id = 'HuggingFaceH4/zephyr-7b-beta', tools = ["google-serper", "llm-math", "serpapi"]):
    assert model_id in {'text-davinci-003',
                        'meta-llama/Llama-2-70b-chat-hf',
                        'mistralai/Mixtral-8x7B-Instruct-v0.1',
                        'OpenHermes-2.5-Mistral-7B',
                        'HuggingFaceH4/zephyr-7b-beta',
                        'SOLAR-10.7B-Instruct-v1.0'}
    if model_id == 'text-davinci-003':
      llm = OpenAI(model_name = model_id, temperature = 0)
    else:
      llm = HuggingFaceEndpoint(repo_id = model_id)
    tools = load_tools(tools, llm = llm)
    self.agent = initialize_agent(tools, llm, agent = "zero-shot-react-description", verbose = True)
  def query(self, question):
    self.run(question)

