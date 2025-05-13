import torch
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from collections import Counter

# Load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.eval()

def detect_objects(image):
    image_np = np.array(image)
    results = model(image_np)
    results.render()
    result_img = Image.fromarray(results.ims[0])
    
    labels = results.pandas().xyxy[0]['name'].tolist()
    label_counts = Counter(labels)
    label_summary = ', '.join([f"{v} {k}(s)" for k, v in label_counts.items()]) or "No objects detected"

    return result_img, label_summary

# Custom CSS
custom_css = """
body { background-color: #111827; color: white; font-family: 'Poppins', sans-serif; }
.gradio-container { background-color: #1f2937 !important; border-radius: 20px; padding: 2rem; }
h1, h2, h3, h4, h5, h6 { color: #facc15; }
label { font-weight: bold; }
textarea, input { background-color: #374151 !important; color: white !important; border: 1px solid #6b7280 !important; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
    gr.Markdown("##  YOLOv5 Object Detection App")
    gr.Markdown("Upload or capture an image to detect objects using **YOLOv5**. Results will be shown below 👇")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload or Click Image")
            btn = gr.Button(" Detect Objects")
        with gr.Column():
            output_img = gr.Image(type="pil", label=" Detected Image")
            summary = gr.Textbox(label="🧾 Object Summary")

    btn.click(fn=detect_objects, inputs=input_img, outputs=[output_img, summary])

demo.launch()
