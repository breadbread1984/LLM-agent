#!/usr/bin/python3

from os import environ
from getpass import getpass
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceEndpoint, OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools

class Agent(object):
  def __init__(self, model = 'zephyr', tools = ["google-serper", "llm-math"]):
    # openai is not available in china, cannot singup for an api key
    # get token from https://platform.openai.com/overview
    environ['OPENAI_API_KEY'] = 'to be filled'
    environ['OPENAI_ORGANIZATION'] = 'HKQAI'
    # get token from https://huggingface.co/
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    # get token from https://serper.dev/
    environ["SERPAPI_API_KEY"] = 'd075ad1b698043747f232ec1f00f18ee0e7e8663'
    openai_model_list = {'ada001': 'text-ada-001',
                         'babbage001': 'text-babbage-001',
                         'curie001': 'text-curie-001',
                         'davinci001': 'text-davinci-001',
                         'davinci002': 'text-davinci-002',
                         'davinci003': 'text-davinci-003',
                         'gpt3.5': 'gpt-3.5-turbo-instruct',
                         'gpt4': 'gpt-4'}
    hf_model_list = {'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                     'mistral': 'teknium/OpenHermes-2.5-Mistral-7B',
                     'solar': 'upstage/SOLAR-10.7B-Instruct-v1.0',
                     'llama2': 'meta-llama/Llama-2-7b-chat-hf',
                     'zephyr': "HuggingFaceH4/zephyr-7b-beta",}
    if model in openai_model_list:
      llm = OpenAI(model_name = openai_model_list[model],
                   temperature = 0,
                   openai_api_key = getpass(),
                   openai_organization = 'to be filled')
    elif model in hf_model_list:
      llm = HuggingFaceEndpoint(repo_id = hf_model_list[model],
                                token = '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')
    else:
      raise Exception('unknown model!')
    memory = ConversationBufferMemory(memory_key="chat_history")
    tools = load_tools(tools, llm = llm, serper_api_key = 'd075ad1b698043747f232ec1f00f18ee0e7e8663')
    self.agent_chain = initialize_agent(tools = tools, llm = llm, memory = memory, agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose = True)
  def query(self, question):
    return self.agent_chain.run(question)

if __name__ == "__main__":
  agent = Agent(model = 'zephyr')
  print(agent.query("what is SwiGLU?"))
