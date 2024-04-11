#!/usr/bin/python3

from os import environ
import json
import logging
from huggingface_hub import login
from langchain import hub
from langchain.llms import HuggingFaceEndpoint, OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.agents import initialize_agent, AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain_community.utilities import SerpAPIWrapper
from models import ChatGLM3, Llama2, Zephyr

class Agent(object):
  def __init__(self, model = 'zephyr', tools = ["llm-math", "serpapi"], device = 'cuda'):
    assert device in {'cpu', 'cuda'}
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    hf_model_list = {'chatglm3': 'THUDM/chatglm3-6b',
                     'llama2': 'meta-llama/Llama-2-7b-chat-hf',
                     'zephyr': "HuggingFaceH4/zephyr-7b-beta",}
    environ["SERPAPI_API_KEY"] = '052741ba82a96e21b3c3ab35e6c5288f470a11402bc83a9cc86c306f826d24f0'
    if model == 'openai':
      llm = OpenAI(model_name = 'text-davinci-003', temperature = 0)
    else:
      llm = HuggingFaceEndpoint(repo_id = hf_model_list[model], token = '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')
    tools = load_tools(tools, llm = llm, serper_api_key = '052741ba82a96e21b3c3ab35e6c5288f470a11402bc83a9cc86c306f826d24f0')
    self.agent_chain = initialize_agent(tools = tools, llm = llm, agent = "zero-shot-react-description", verbose = True)
  def query(self, question):
    return self.agent_chain.run(question)

if __name__ == "__main__":
  agent = Agent(model = 'zephyr', device = 'cpu')
  print(agent.query("where does the word maelstrom come?"))
