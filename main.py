#!/usr/bin/python3

import time
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
  def query(self, history):
    bot_message = self.agent.query(history[-1][0])
    history[-1][1] = ""
    for character in bot_message:
      history[-1][1] += str(character)
      time.sleep(0.03)
      yield history

def main(unused_argv):
  warper = Warper(FLAGS.model)
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
      submit_btn.click(lambda user_message, history: ("", history + [[user_message, None]]), inputs = [msg, chatbot], outputs = [msg, chatbot], queue = False).then(warper.query, chatbot, chatbot)
  gr.close_all()
  demo.launch(server_name = FLAGS.host, server_port = FLAGS.port)

if __name__ == "__main__":
  add_options()
  app.run(main)
