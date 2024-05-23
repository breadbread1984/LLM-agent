#!/usr/bin/python3

from os import environ, remove
from os.path import exists, join, isdir
from typing import Optional, Type
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.graphs import Neo4jGraph
from langchain_community.llms import HuggingFaceEndpoint
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from prompts import entity_generation_template, triplets_qa_template
from reaction_path import PrecursorsRecommendation

def load_knowledge_graph(host = 'bolt://localhost:7687', username = 'neo4j', password = None, database = 'neo4j'):
  class ChemKnowledgeInput(BaseModel):
    query: str = Field(description = "should be a search query")

  class ChemKnowledgeConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    neo4j: Neo4jGraph
    tokenizer: PreTrainedTokenizerFast
    llm: HuggingFaceEndpoint

  class ChemKnowledgeTool(BaseTool):
    name = "Chemistry Knowledge Graph"
    description = 'userful when you need to answer questions about chemistry'
    args_schema: Type[BaseModel] = ChemKnowledgeInput
    return_direct: bool = True
    config: ChemKnowledgeConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
      # 1) extract entities of known entity types
      results = self.config.neo4j.query("match (n) return distinct labels(n)")
      entity_types = [result['labels(n)'][0] for result in results]
      prompt = entity_generation_template(self.config.tokenizer, entity_types)
      chain = prompt | self.config.llm
      entities = chain.invoke({'question': query})
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
      answer = chain.invoke({'question': query})
      return answer
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
      raise NotImplementedError("Chemistry Knowledge Graph does not suppert async!")

  environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
  neo4j = Neo4jGraph(url = host, username = username, password = password, database = database)
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
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
  return ChemKnowledgeTool(config = ChemKnowledgeConfig(neo4j = neo4j, tokenizer = tokenizer, llm = llm))

def load_precursor_predictor(device = 'cpu'):
  import gdown
  import zipfile
  if not exists('rsc') or not isdir('rsc'):
    url = 'https://drive.google.com/uc?id=1gkZyU6TQI6c4m9lLPM2J7a9GLwRur7N6'
    zip_path = 'rsc.zip'
    gdown.download(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as f:
      f.extractall('.')
    remove(zip_path)
  if not exists('reaction_path_ckpt') or not isdir('reaction_path_ckpt'):
    url = 'https://drive.google.com/uc?id=1dDiCcWNEbsnPyKrZYXsYiOsWiLAsmii3'
    zip_path = 'reaction_path_ckpt.zip'
    gdown.download(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as f:
      f.extractall('.')
    remove(zip_path)

  class PrecursorInput(BaseModel):
    compound: str = Field(description = "should be the chemical formula of a compound")
    count: int = Field(description = "should be the number of groups of precursors")

  class PrecursorTool(BaseTool):
    name = "Reaction Precursor Prediction"
    description = 'useful when you want to guess the precursors of a compound in chemical reaction'
    args_schema: Type[BaseModel] = PrecursorInput
    return_direct: bool = True
    recommend: PrecursorsRecommendation = PrecursorsRecommendation(model_dir = 'reaction_path_ckpt', data_dir = 'rsc', device = device)
    def _run(self, compound: str, count: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> list:
      target_formula = [compound]
      all_predicts = self.recommend.call(target_formula = target_formula, top_n = count)
      return all_predicts[0]['precursors_predicts']
    async def _arun(self, compound: str, count: int, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
      raise NotImplementedError("Reaction Precursor Prediction does not suppert async!")

  return PrecursorTool()

if __name__ == "__main__":
  kb = load_knowledge_graph(password = '19841124')
  print('name:',kb.name)
  print('description:', kb.description)
  print('args:',kb.args)
  res = kb.invoke({'query': 'what is the application of sodium chloride?'})
  print(res)
  # NOTE: https://github.com/langchain-ai/langchain/discussions/15927
  kb.config.neo4j._driver.close()
  precursor = load_precursor_predictor()
  print('name:',precursor.name)
  print('description:', precursor.description)
  print('args:', precursor.args)
  res = precursor.invoke({'compound': 'SrZnSO', 'count': 10})
  print(res)
