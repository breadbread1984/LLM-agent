#!/usr/bin/python3

from os import environ
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceEndpoint, OpenAI
from langchain.agents import initialize_agent, load_tools

class Agent(object):
  def __init__(self, model = 'zephyr', tools = ["google-serper", "llm-math"], device = 'cuda'):
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
    memory = ConversationBufferMemory(memory_key="chat_history")
    tools = load_tools(tools, llm = llm, serper_api_key = '052741ba82a96e21b3c3ab35e6c5288f470a11402bc83a9cc86c306f826d24f0')
    self.agent_chain = initialize_agent(tools = tools, llm = llm, memory = memory, agent = "zero-shot-react-description", verbose = True)
  def query(self, question):
    return self.agent_chain.run(question)

if __name__ == "__main__":
  agent = Agent(model = 'zephyr', device = 'cpu')
  print(agent.query("where does the word maelstrom come?"))
