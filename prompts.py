#!/usr/bin/python3

from langchain import hub
from langchain.tools.render import render_text_description
from langchain_core.prompts.prompt import PromptTemplate

def agent_template(tokenizer, tools):
  prompt = hub.pull('hwchase17/react-json')
  system_template = prompt[0].prompt.template
  system_template = system_template.replace('{tools}', render_text_description(tools))
  system_template = system_template.replace('{tool_names}', ", ".join([t.name for t in tools]))
  user_template = prompt[1].prompt.template
  messages = [
    {'role': 'system', 'content': system_template},
    {'role': 'user', 'content': user_template}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['agent_scratchpad', 'input'])
  return template

def entity_generation_template(tokenizer, entity_types):
  entity_template = "Extract entities of type among %s from the user given question. Focus on extracting the entities that we can use to best lookup answers to the question. Provide entities sorted by their types in the following format: {'type1': [entity1, entity2], 'type2': [], 'type3': ['entity3']}. Reply no extra words besides entities." % str(entity_types)
  entity_template = entity_template.replace('{','{{')
  entity_template = entity_template.replace('}','}}')
  messages = [
    {'role': 'system', 'content': entity_template},
    {'role': 'user', 'content': "{question}"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['question'])
  return template

def triplets_qa_template(tokenizer, triplets):
  qa_template = "Base only on triplets extracted from knowledge graph as fact, please reply user's question. If the triplets gives no clue to the question, just answer 'I can't answer your question.'. Triplets: %s" % str(triplets)
  qa_template = qa_template.replace('{','{{')
  qa_template = qa_template.replace('}','}}')
  messages = [
    {'role': 'system', 'content': qa_template},
    {'role': 'user', 'content': "{question}"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['question'])
  return template

