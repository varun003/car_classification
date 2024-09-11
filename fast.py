from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
from huggingface_hub import hf_hub_download

app = FastAPI()

def load_model_from_hf(repo_id, filename):
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = tf.keras.models.load_model(model_path)
    return model

# Load model from Hugging Face
model = load_model_from_hf('varun-negi/car_classification', 'car_classification_model.h5')


# # Load your trained model
# model = tf.keras.models.load_model('car_classification_model.h5')  ## car_classification_model_8bit.tflite # car_classification_model.h5

def preprocess_image(image: BytesIO):
    image = Image.open(image).convert('RGB')
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()
        image = preprocess_image(BytesIO(image))

        # Make predictions
        brand_predictions, model_predictions = model.predict(image)

        # Convert predictions to class labels
        brand_label = np.argmax(brand_predictions, axis=1)[0]
        model_label = np.argmax(model_predictions, axis=1)[0]

        return JSONResponse(content={
            'brand': int(brand_label),
            'model': int(model_label)
        })
    except Exception as e:
        return JSONResponse(content={
            'error': str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


## uvicorn main:app --reload