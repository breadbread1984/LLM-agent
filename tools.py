#!/usr/bin/python3

from os import environ
from typing import Optional, Type
from transformers import AutoTokenizer
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.graphs import Neo4jGraph
from langchain_community.llms import HuggingFaceEndpoint
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from prompts import entity_generation_template, triplets_qa_template

class ChemKnowledgeInput(BaseModel):
  query: str = Field(description = "should be a search query")

class ChemKnowledgeConfig(BaseModel):
  neo4j: Neo4jGraph
  tokenizer: AutoTokenizer
  llm: HuggingFaceEndpoint

class ChemKnowledgeTool(BaseTool):
  name = "Chemistry Knowledge Graph"
  description = 'userful for when you need to answer questions about chemistry'
  args_schema: Type[BaseModel] = ChemKnowledgeInput
  return_direct: bool = True
  config: ChemKnowledgeConfig
  def __init__(self, config: ChemKnowledgeConfig):
    self.config = config
  def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
    # 1) extract entities of known entity types
    results = self.config.neo4j.query("match (n) return distinct labels(n)")
    entity_types = [result['labels(n)'][0] for result in results]
    prompt = entity_generation_template(self.config.tokenizer, entity_types)
    chain = prompt | self.config.llm
    entities = chain.invoke({'question': question})
    entities = eval(entities)
    print('extracted entityes:', entities)
    # 2) search for triplets related to these triplets
    triplets = list()
    for entity_type, keywords in entities.items():
      if len(keywords) == 0: continue
      for keyword in keywords:
        #cypher_cmd = 'match (a:`%s`)-[r]->(b) where tolower(a.id) contains tolower(\'%s\') return a,r,b' % (entity_type, keyword)
        cypher_cmd = 'match (a)-[r]->(b) where tolower(a.id) contains tolower(\'%s\') return a,r,b' % (keyword)
        matches = self.config.neo4j.query(cypher_cmd)
        triplets.extend([(match['a']['id'],match['r'][1],match['b']['id']) for match in matches])
        #cypher_cmd = 'match (b)-[r]->(a:`%s`) where tolower(a.id) contains tolower(\'%s\') return b,r,a' % (entity_type, keyword)
        cypher_cmd = 'match (b)-[r]->(a) where tolower(a.id) contains tolower(\'%s\') return b,r,a' % (keyword)
        matches = self.config.neo4j.query(cypher_cmd)
        triplets.extend([(match['b']['id'],match['r'][1],match['a']['id']) for match in matches])
    print('matched triplets:', triplets)
    # 3) ask llm for answer according to matched triplets
    prompt = triplets_qa_template(self.config.tokenizer, triplets)
    chain = prompt | self.config.llm
    answer = chain.invoke({'question': question})
    return answer
  async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
    raise NotImplementedError("Chemistry Knowledge Graph does not suppert async!")

if __name__ == "__main__":
  environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
  config = ChemKnowledgeConfig(
    neo4j = Neo4jGraph(url = 'bolt://localhost:7687', username = 'neo4j', password = '19841124', database = 'neo4j'),
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct'),
    llm = HuggingFaceEndpoint(
      endpoint_url = "meta-llama/Meta-Llama-3-8B-Instruct",
      task = "text-generation",
      max_length = 16384,
      do_sample = False,
      temperature = 0.6,
      top_p = 0.9,
      eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
      use_cache = True,
    )
  )
  kb = ChemKnowledgeTool(config = config)
  res = kb('what is the application of sodium chloride?')
  print(res)
