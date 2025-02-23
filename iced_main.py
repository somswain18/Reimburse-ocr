from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd
from IPython.display import display, Image
from langchain_core.runnables.graph import MermaidDrawMethod
from dotenv import load_dotenv
import os
import uuid
from pydantic import BaseModel, Field
from typing import List, TypedDict
from urllib.request import urlopen
import json
import requests
from datetime import datetime, timedelta
from langgraph.graph import StateGraph, START,END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import pytesseract
from langchain_groq import ChatGroq
import io
from PIL import Image
import numpy as np
from IPython.display import display, Image as PILImage
import gradio as gr



load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Defining the AI agent class
class Billtype(BaseModel):
    typee: int = Field(description="List of BILLS")
from typing import TypedDict

class ReimburesemtAgent(TypedDict):
  the_image: bytes
  extracted_text: str
  extracted_price: str
  extracted_bill_type: int
  unique_id: uuid.UUID
  name:str
  
  
  
  
'''
REPLACE THE PATHS TO YOUR OWN PERSONAL DB OR DRIVE.

'''
# Create a csv file for your user

def creating_new_name(name):
  data = {
      'image': [],
      'extract_text': [],
      'extract_price': [],
      'extract_type': []
  }
  df = pd.DataFrame(data)
  df.to_csv(f'/content/drive/MyDrive/reimbursement/storage/{name}.csv',index=False)
creating_new_name('sumit')
  
  
  
# Defining OCR and Agent functions
def run_tesseract(state:ReimburesemtAgent):



    img_byte_arr=state['the_image']
    img = Image.open(io.BytesIO(img_byte_arr))

    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    text = pytesseract.image_to_string(img)
    new_uuid =uuid.uuid4()
    # del state['the_image']
    return {'extracted_text':text,'unique_id':new_uuid}

def extract_price(state:ReimburesemtAgent):
    text = state['extracted_text']
    prompt = f'''I have extracted text from an OCR model of a bill or receipt of an expense I want to reimburese,
    I want you to tell me the exact total value of the bill.

    return me just a number nothing else


    The extracted text is :{text}
    '''

    response = llm.invoke(prompt)

    return {'extracted_price':response.content}

def extract_type(state:ReimburesemtAgent):
    text = state['extracted_text']
    prompt = f'''I have extracted text from an OCR model of a bill or receipt of an expense I want to reimburese,
    I want you to tell me what type of bill it is.

    return me just the type from the fillowing types:
    1) Fuel
    2) Travel
    3) Hotel
    4) Food
    5) others



    The extracted text is :{text}
    '''
    structure_llm = llm.with_structured_output(Billtype)
    bill: Billtype = structure_llm.invoke(prompt)
    bill.typee

    return {'extracted_bill_type':bill.typee}

def saving_in_drive(state:ReimburesemtAgent):



    img = Image.open(io.BytesIO(state['the_image']))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()


    extract_text=state['extracted_text']
    extract_price=state['extracted_price']
    extract_type=state['extracted_bill_type']
    idd=state['unique_id']
    name = state['name']


    if int(extract_type) == 1:
      extract_type='Fuel'
    elif int(extract_type) == 2:
      extract_type='Travel'
    elif int(extract_type) == 3:
      extract_type='Hotel'
    elif int(extract_type) == 4:
      extract_type='Food'
    else:
      extract_type='other'
    # reading file
    df = pd.read_csv(f'/content/drive/MyDrive/reimbursement/storage/{name}.csv')
    new_row = pd.DataFrame([[img_byte_arr, extract_text, extract_price, extract_type]],
                      columns=['image', 'extract_text', 'extract_price', 'extract_type'])
    # saving new file
    df = pd.concat([df, new_row], ignore_index=True)
    # df = df.sort_values(by='extract_price', ascending=False)
    df.to_csv(f'/content/drive/MyDrive/reimbursement/storage/{name}.csv',index=False)


# Defining LangGraph


workflow = StateGraph(ReimburesemtAgent)


workflow.add_node("Running_OCR_for_text", run_tesseract)
workflow.add_node("Extract_price", extract_price)
workflow.add_node("Extract_type", extract_type)
workflow.add_node("save_in_drive", saving_in_drive)


workflow.set_entry_point("Running_OCR_for_text")
workflow.add_edge("Running_OCR_for_text", "Extract_price")
workflow.add_edge("Running_OCR_for_text", "Extract_type")

workflow.add_edge("Extract_price", "save_in_drive")
workflow.add_edge("Extract_type", "save_in_drive")
workflow.add_edge("save_in_drive", END)

memory = MemorySaver()

graph_plan = workflow.compile(checkpointer=memory)
display(Image(graph_plan.get_graph(xray=1).draw_mermaid_png()))


# Getting a Gradio instance up
config = {"configurable": {"thread_id": "1"}}
from PIL import Image

state_input = {
        "the_image": bytearray(),
        "extracted_text": "",
        "extracted_price": "",
        "extracted_bill_type": "",
        "unique_id": "",
        "name":"chirag"
}

def process_image(image, name):

  img = Image.open(image)
  img_byte_arr = io.BytesIO()
  img.save(img_byte_arr, format='JPEG')
  img_byte_arr = img_byte_arr.getvalue()
  state_input["the_image"] = img_byte_arr
  for event in graph_plan.stream(state_input, config, stream_mode=["updates"]):
       print(f"Current node: {next(iter(event[1]))}")

  try:
    df = pd.read_csv(f'/content/drive/MyDrive/reimbursement/storage/{name}.csv')
    return df[['extract_price','extract_type']].to_html()
  except FileNotFoundError:
    return "Error: No such file found."

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="filepath"), # Changed to filepath for image upload
        gr.Textbox(label="Your Name")
    ],
    outputs=gr.HTML(),
    title="Image Processor",
    description="Upload an image, it will be processed and the associated data from excel file will be shown."
)

iface.launch()
