#!/usr/bin/python3

from absl import flags, app
import gradio as gr
from agent import Agent

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('host', default = '0.0.0.0', help = 'host address')
  flags.DEFINE_integer('port', default = 8880, help = 'port number')
  flags.DEFINE_enum('model', default = 'zephyr', enum_values = {'ada001', 'babbage001', 'curie001', 'davinci001', 'davinci002', 'davinci003', 'gpt3.5', 'gpt4', 'mixtral', 'mistral', 'solar', 'llama2', 'zephyr'}, help = 'model to use')

class Warper(object):
  def __init__(self, model):
    self.agent = Agent(model)
  def query(self, question):
    return self.agent.query(question)

def main(unused_argv):
  warper = Warper()
  block = gr.Blocks()
  with block as demo:
    with gr.Row(equal_height = True):
      with gr.Column(scale = 15):
        gr.Markdown("<h1><center>LLM Agent</center></h1>")
      with gr.Row():
        with gr.Column(scale = 4):
          chatbot = gr.Chatbot(height = 450, show_copy_button = True)
          msg = gr.Textbox(label = '需要问什么？')
          with gr.Row():
            submit_btn = gr.Button("发送")
          with gr.Row():
            clear_btn = gr.ClearButton(components = [chatbot], value = "清空问题")
        submit_btn.click(warper.query, input = [msg, chatbot], outputs = [msg, chatbot])
  gr.close_all()
  demo.launch(server_name = FLAGS.host, server_port = FLAGS.port)

if __name__ == "__main__":
  add_options()
  app.run(main)
