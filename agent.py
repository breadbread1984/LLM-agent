#!/usr/bin/python3

from os import environ
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceEndpoint, OpenAI
from langchain.agents import initialize_agent, load_tools

class Agent(object):
  def __init__(self, model = 'zephyr', tools = ["google-serper", "llm-math"], device = 'cuda'):
    assert device in {'cpu', 'cuda'}
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    environ["SERPAPI_API_KEY"] = 'd075ad1b698043747f232ec1f00f18ee0e7e8663'
    hf_model_list = {'chatglm3': 'THUDM/chatglm3-6b',
                     'llama2': 'meta-llama/Llama-2-7b-chat-hf',
                     'zephyr': "HuggingFaceH4/zephyr-7b-beta",}
    if model == 'openai':
      llm = OpenAI(model_name = 'text-davinci-003', temperature = 0)
    else:
      llm = HuggingFaceEndpoint(repo_id = hf_model_list[model], token = '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')
    memory = ConversationBufferMemory(memory_key="chat_history")
    tools = load_tools(tools, llm = llm, serper_api_key = 'd075ad1b698043747f232ec1f00f18ee0e7e8663')
    self.agent_chain = initialize_agent(tools = tools, llm = llm, memory = memory, agent = "zero-shot-react-description", verbose = True)
  def query(self, question):
    return self.agent_chain.run(question)

if __name__ == "__main__":
  agent = Agent(model = 'zephyr', device = 'cpu')
  print(agent.query("list famous actors and actress of hong kong."))
