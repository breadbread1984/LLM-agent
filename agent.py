#!/usr/bin/python3

from langchain.llms import HuggingFaceEndpoint
from langchain.chat_models.huggingface import ChatHuggingFace

def get_agent(model_id):
  llm = HuggingFaceEndpoint(repo_id = model_id)
  chat_model = ChatHuggingFace(llm = llm)
  return chat_model

