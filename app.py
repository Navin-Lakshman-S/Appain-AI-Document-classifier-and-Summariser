from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
import time
import os
import pdf2image as pdi
from dotenv import load_dotenv
from openai import OpenAI
import base64
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pickle

# Initialize Flask app
app = Flask(__name__)
load_dotenv()

# Load the pre-trained model and tokenizer
with open('Tokenizer.pickle', 'rb') as h:
    tokenizer = pickle.load(h)

with open('LabelEncoder.pickle', 'rb') as l:
    label_encoder = pickle.load(l)
# print(type(label_encoder))
loaded_model = load_model('lstm_keras_model.keras')
print("Model loaded successfully.")


def encode(image):
    with open(image, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
def t2i(image):
    bs64img = encode(image)

    client = OpenAI(
        api_key=os.getenv('XAI'),
        base_url="https://api.x.ai/v1",
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{bs64img}",
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": "You are an OCR model. Your task is to extract only the English words from the attached image. Return only the extracted text. Do not provide any context, explanations, or other commentary.",
                },
            ],
        },
    ]

    completion = client.chat.completions.create(
        model="grok-2-vision-1212",
        messages=messages,
        stream=False,
        temperature=0.01,
    )

    return completion.choices[0].message.content    


# Function to preprocess input text and predict label
def predict_label(input_text):
    # Preprocess the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequence, maxlen=100)

    # Predict the label
    prediction = loaded_model.predict(input_padded)
    predicted_class = np.argmax(prediction, axis=1)[0]
    label = label_encoder.inverse_transform([predicted_class])[0]

    return label

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_images = pdi.convert_from_path(pdf_file, dpi=300)
    dir_path = os.getcwd()
    image_dir = os.path.join(dir_path, 'pdf_images')
    os.makedirs(image_dir, exist_ok=True)
    all_text = ""
    
    for i, img in enumerate(pdf_images):
        image_path = os.path.join(image_dir, f'page_{i}_{time.time()}.jpg')
        img.save(image_path, 'JPEG')
        result = t2i(image_path)
        # page_text = ' '.join([text[1] for text in result])
        all_text += result + '\n'
    print("Txt Extrt")
    print(all_text)
    print("Txt extrt done")
    return all_text

# Route to home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle PDF upload and classification
@app.route('/classify', methods=['POST'])
def classify():
    if 'pdf_file' not in request.files:
        return redirect(request.url)
    
    pdf_file = request.files['pdf_file']
    
    if pdf_file.filename == '':
        return redirect(request.url)
    
    if pdf_file:
        # Save the uploaded PDF file
        pdf_path = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(pdf_path, exist_ok=True)
        pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', pdf_file.filename)
        pdf_file.save(pdf_path)

        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_path)

        print("Txt Extrt")
        print(extracted_text)
        print("Txt extrt done")
    
        # Predict the label using the extracted text
        predicted_label = predict_label(extracted_text)

        print(predicted_label)

        if predicted_label == 0:
            document_type = "Application Document"
        elif predicted_label == 1:
            document_type = "Identity Document"
        elif predicted_label == 2:
            document_type = "Financial Document"
        elif predicted_label == 3:
            document_type = "Receipt"
        else:
            document_type = "None"

        return render_template('index.html', document_type=document_type)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
