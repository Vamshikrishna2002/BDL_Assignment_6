import uvicorn
from fastapi import FastAPI, UploadFile, File
from keras.models import load_model
from predict import predict_digit
from io import BytesIO

import numpy as np
from PIL import Image

model=load_model("MNIST_Model.h5")
app = FastAPI()

def format_image(image):
    image = image.resize((28, 28)).convert('L')
    image_array = np.array(image).reshape(784)
    return image_array


@app.post('/predict')
async def predict(upload_file: UploadFile = File(...)):
    
    contents = await upload_file.read()
    image = Image.open(BytesIO(contents))
    image_array = format_image(image)
    print(image_array.max())
    digit = predict_digit(model, image_array)
    return {"digit": digit}