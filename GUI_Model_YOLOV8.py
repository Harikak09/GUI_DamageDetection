import gradio as gr
import pandas as pd
from ultralytics import YOLO
import os
from PIL import Image, ImageDraw

# Load your YOLO model
model_path = r"C:\Users\harik\Downloads\GUI\best.pt"
yolo_model = YOLO(model_path)

# Function to predict and visualize bounding boxes
def predict_and_visualize(image):
    """
    Runs the YOLO model on the uploaded image, 
    predicts bounding boxes, and returns the image with bounding boxes.
    """
    # Run prediction
    results = yolo_model(image)
    result = results[0]  # First result (in case of batch)
    print(result.boxes.data)
    # Load the original image
    img = Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw bounding boxes
    for box in result.boxes.data.tolist():  # Get bounding boxes
        x1, y1, x2, y2, score, detected_class = box
        label = f"Prob of {detected_class}: ({score:.2f})"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)  # Bounding box
        draw.text((x1, y1), label, fill="blue")  # Label

    return img

# Define the Gradio interface
def gradio_interface():
    """
    Creates the Gradio interface.
    """
    # Create Gradio interface
    interface = gr.Interface(
        fn=predict_and_visualize,  # Function to call
        inputs=gr.Image(type="filepath", label="Upload an Image"),  # Input: Image
        outputs=gr.Image(type="pil", label="Image with Bounding Boxes"),  # Output: Image with boxes
        title="YOLOv8 Object Detection",
        description="Detect the damaged roads"
    )
    return interface

# Launch the Gradio app
if __name__ == "__main__":
    app = gradio_interface()
    app.launch()
