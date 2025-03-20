import os
import uuid
import io
import pandas as pd
from PIL import Image
import pytesseract
import gradio as gr
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import TypedDict

import os

# Define local storage path
STORAGE_DIR = "/home/som/Documents/reimburse_storage"

# Ensure the directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)


# Load API Key from Environment
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=None, timeout=None, max_retries=2)

# Define local storage path
STORAGE_DIR = "/home/som/Documents/reimburse_storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

# Define BillType Model
class Billtype(BaseModel):
    typee: int = Field(description="List of BILLS")

class ReimbursementAgent(TypedDict):
    the_image: bytes
    extracted_text: str
    extracted_price: str
    extracted_bill_type: int
    unique_id: uuid.UUID
    name: str

# Function to create a new user file
def create_user_file(name):
    file_path = os.path.join(STORAGE_DIR, f"{name}.csv")
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=['image_path', 'extract_text', 'extract_price', 'extract_type'])
        df.to_csv(file_path, index=False)

# OCR Extraction
def run_tesseract(state: ReimbursementAgent):
    img = Image.open(io.BytesIO(state['the_image']))
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    text = pytesseract.image_to_string(img)
    return {'extracted_text': text, 'unique_id': uuid.uuid4()}

# Extract Price
def extract_price(state: ReimbursementAgent):
    prompt = f"""Extract the total bill amount from the following OCR text:

    {state['extracted_text']}
    
    Return only a number."""
    response = llm.invoke(prompt)
    return {'extracted_price': response.content.strip()}

# Extract Bill Type
def extract_type(state: ReimbursementAgent):
    prompt = f"""Classify the bill type from the OCR text:

    {state['extracted_text']}

    Choose from: 1) Fuel, 2) Travel, 3) Hotel, 4) Food, 5) Other"""
    structure_llm = llm.with_structured_output(Billtype)
    bill: Billtype = structure_llm.invoke(prompt)
    return {'extracted_bill_type': bill.typee}

# Save to Local Storage
def save_to_local(state: ReimbursementAgent):
    name = state['name']
    file_path = os.path.join(STORAGE_DIR, f"{name}.csv")
    image_name = f"{state['unique_id']}.jpg"
    image_path = os.path.join(STORAGE_DIR, image_name)
    
    with open(image_path, 'wb') as f:
        f.write(state['the_image'])
    
    bill_type_map = {1: 'Fuel', 2: 'Travel', 3: 'Hotel', 4: 'Food', 5: 'Other'}
    extract_type = bill_type_map.get(state['extracted_bill_type'], 'Other')
    
    df = pd.read_csv(file_path)
    new_row = pd.DataFrame([[image_path, state['extracted_text'], state['extracted_price'], extract_type]],
                           columns=['image_path', 'extract_text', 'extract_price', 'extract_type'])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file_path, index=False)

# Define LangGraph Workflow
workflow = StateGraph(ReimbursementAgent)
workflow.add_node("Running_OCR_for_text", run_tesseract)
workflow.add_node("Extract_price", extract_price)
workflow.add_node("Extract_type", extract_type)
workflow.add_node("save_to_local", save_to_local)
workflow.set_entry_point("Running_OCR_for_text")
workflow.add_edge("Running_OCR_for_text", "Extract_price")
workflow.add_edge("Running_OCR_for_text", "Extract_type")
workflow.add_edge("Extract_price", "save_to_local")
workflow.add_edge("Extract_type", "save_to_local")
workflow.add_edge("save_to_local", END)
workflow.compile()

# Gradio Interface
def process_image(image, name):
    create_user_file(name)
    
    img = Image.open(image)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    state_input = {
        "the_image": img_byte_arr,
        "extracted_text": "",
        "extracted_price": "",
        "extracted_bill_type": "",
        "unique_id": "",
        "name": name
    }
    
    for event in workflow.stream(state_input):
        print(f"Current node: {next(iter(event[1]))}")
    
    df = pd.read_csv(os.path.join(STORAGE_DIR, f"{name}.csv"))
    return df[['extract_price', 'extract_type']].to_html()

iface = gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="filepath"), gr.Textbox(label="Your Name")],
    outputs=gr.HTML(),
    title="Image Processor",
    description="Upload an image to process and retrieve bill details."
)

iface.launch()
