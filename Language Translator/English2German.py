#Installing Dependencies
pip install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install transformers ipywidgets gradio --upgrade

#Importing
import gradio as gr                   #UI Library
from transformers import pipeline    #Tranformers Pipeline

#Translating Function
def translate_transformers(Input):
  translation_pipeline = pipeline('translation_en_to_de')
  results = translation_pipeline(Input)
  return results[0]['translation_text']

interface = gr.Interface(translate_transformers,
                         inputs=gr.inputs.Textbox(lines=2,placeholder='Text to translate'),
                         outputs='text')
interface.launch()
