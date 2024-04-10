#!/usr/bin/python3

from langchain import hub
from langchain.llms import HuggingFaceEndpoint, OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain_community.utilities import SerpAPIWrapper

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
      chat_model = ChatOpenAI(llm = llm)
    else:
      llm = HuggingFaceEndpoint(repo_id = model_id)
      chat_model = ChatHuggingFace(llm = llm)
    tools = load_tools(tools, llm = llm)
    prompt = hub.pull("hwchase17/react-json")
    prompt = prompt.partial(tools = render_text_description(tools), tool_names = ", ".join([t.name for t in tools]))
    agent = {"input": lambda x: x["input"], "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"])} | prompt | chat_model.bind(stop = ["\nObservation"]) | ReActJsonSingleInputOutputParser()
    self.agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)
  def query(self, question):
    return self.aget_executor.invoke({"input": question})

if __name__ == "__main__":
  agent = Agent()
  print(agent.query("who is Jinping Xi's daughter?"))
