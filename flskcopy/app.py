from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_file
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import google.generativeai as genai
from keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import base64
import gridfs




# Configuration
genai.configure(api_key="AIzaSyCldpTHPZ5aCHKG-lcnBWTDIs8F8J7BDyM")


# MongoDB configuration
uri = "mongodb+srv://prabhjeetdec03:N7O1VfinSE7wJNKg@flaskcluster.rayqc.mongodb.net/?retryWrites=true&w=majority&appName=FlaskCluster"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['DiagnoPlus']
patients_collection = db['patients']
pdf_collection = db['pdfs']


fs = gridfs.GridFS(db)

#test mongodb connection 
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


app = Flask(__name__)
CORS(app)

# Load models
breast_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=3)
breast_model.load_state_dict(torch.load("vit_breast_cancer_classifier.pth", map_location=torch.device('cpu'), weights_only=True))
breast_model.eval()

skin_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=2)
skin_model.load_state_dict(torch.load("vit_skin_cancer_classifier.pth", map_location=torch.device('cpu'), weights_only=True))
skin_model.eval()

breast_label_map = {0: 'normal', 1: 'benign', 2: 'malignant'}
skin_label_map = {0: 'malignant', 1: 'benign'}

# Load Keras model
keras_model = load_model('C:\pj\sih\Flsk copy\weights.keras')
pneumonia_model = load_model('C:\pj\sih\Flsk copy\pnmodel.keras')

def predict_image_class(image, model, image_processor, label_map):
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(pixel_values=inputs['pixel_values'])
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return label_map[predicted_class]

def predict_tumor(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    img = cv2.resize(img, (150, 150))
    img_array = np.array(img).reshape(1, 150, 150, 3)
    predictions = keras_model.predict(img_array)
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    prediction = labels[predictions.argmax()]
    return prediction

def predict_pneumonia(image):
    # Resize the image to 150x150 as required by the model
    img = image.resize((150, 150))
    
    # Convert the image to an array and rescale it
    img_array = img_to_array(img) / 255.0
    
    # Expand the dimensions to match the input shape (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict the class using the pneumonia model
    prediction = pneumonia_model.predict(img_array)
    predicted_label = 'Pneumonia' if prediction[0] > 0.5 else 'Normal'
    
    return predicted_label

    
# Retrieve PDF from GridFS
@app.route('/api/get-pdf/<patient_id>', methods=['GET'])
def get_pdf(patient_id):
    try:
        patient = patients_collection.find_one({'patientId': patient_id})
        if not patient or 'pdf_id' not in patient:
            return jsonify({'error': 'PDF not found'}), 404

        # Retrieve the PDF from GridFS
        pdf_id = patient['pdf_id']
        pdf_file = fs.get(pdf_id)

        # Return the PDF as a file response
        return send_file(BytesIO(pdf_file.read()), 
                         download_name=f"{patient['name']}_report.pdf",
                         as_attachment=True,
                         mimetype='application/pdf')
    except Exception as e:
        print(f"Error fetching PDF: {e}")
        return jsonify({'error': str(e)}), 500


# Retrieve image from GridFS
@app.route('/api/get-image/<patient_id>', methods=['GET'])
def get_image(patient_id):
    try:
        patient = patients_collection.find_one({'patientId': patient_id})
        if not patient or 'image_id' not in patient:
            return jsonify({'error': 'Image not found'}), 404

        # Retrieve the image from GridFS
        image_id = patient['image_id']
        image_file = fs.get(image_id)

        # Return the image as a file response
        return send_file(BytesIO(image_file.read()), 
                         download_name=f"{patient['name']}_image.jpg",
                         as_attachment=True,
                         mimetype='image/jpeg')
    except Exception as e:
        print(f"Error fetching image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-form-data/<patient_id>', methods=['GET'])
def get_form_data(patient_id):
    try:
        patient = patients_collection.find_one({'patientId': patient_id}, {'_id': 0, 'name': 1, 'age': 1, 'patientId': 1})
        if not patient:
            return jsonify({'error': 'Form data not found'}), 404
        return jsonify(patient)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/breastcancer', methods=['POST'])
def predict_breast_cancer():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image = Image.open(file).convert("RGB")
        predicted_label = predict_image_class(image, breast_model, ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k'), breast_label_map)
        return jsonify({'predicted_class': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/skincancer', methods=['POST'])
def predict_skin_cancer():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image = Image.open(file).convert("RGB")
        predicted_label = predict_image_class(image, skin_model, ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k'), skin_label_map)
        return jsonify({'predicted_class': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tumorprediction', methods=['POST'])
def predict_tumor_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image = Image.open(file).convert("RGB")
        predicted_label = predict_tumor(image)
        return jsonify({'predicted_class': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/disease-facts', methods=['GET'])
def get_disease_facts():
    disease = request.args.get('disease')
    if not disease:
        return jsonify({'error': 'Disease parameter is required'}), 400

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Tell me something about {disease} cancer. if there is a cure for it. best things to do right now dont mention the line you are not doctor just give me the answer for what i asked also dont add * in your answers in three paragraph no special characters should be used in response ")
        facts = [line.strip() for line in response.text.split('\n') if line.strip()]
        return jsonify({'facts': facts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query') + " Don't say you are not a doctor. Just give me the answer for what I asked. Also, don't add * in your answers. Short answer, no special characters should be used in response."
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(query)
        return jsonify({'response': response.text.strip()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return "Test route is working!"


@app.route('/api/pneumonia-prediction', methods=['POST'])
def predict_pneumonia_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        # Open the image
        image = Image.open(file).convert("RGB")
        
        # Predict pneumonia
        predicted_label = predict_pneumonia(image)
        
        # Return the prediction as JSON
        return jsonify({'predicted_class': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/save-pdf', methods=['POST'])
def save_pdf():
    if 'file' not in request.files or 'image' not in request.files or 'name' not in request.form or 'age' not in request.form or 'patientId' not in request.form or 'result' not in request.form:
        return jsonify({'error': 'Missing file, image, form data, or prediction result'}), 400

    pdf_file = request.files['file']
    image_file = request.files['image']
    name = request.form.get('name')
    age = request.form.get('age')
    patient_id = request.form.get('patientId')
    result = request.form.get('result')

    if pdf_file.filename == '' or image_file.filename == '':
        return jsonify({'error': 'No selected PDF or image file'}), 400

    try:
        # Store PDF in GridFS
        pdf_id = fs.put(pdf_file, filename=pdf_file.filename, content_type='application/pdf')

        # Store image in GridFS
        image_id = fs.put(image_file, filename=image_file.filename, content_type='image/jpeg')

        # Create patient data with GridFS file IDs
        patient_data = {
            "name": name,
            "age": age,
            "patientId": patient_id,
            "pdf_id": pdf_id,  # Store GridFS ID for the PDF
            "image_id": image_id,  # Store GridFS ID for the image
            "result": result  # Store the prediction result
        }

        # Insert or update the patient data in MongoDB
        patients_collection.update_one(
            {'patientId': patient_id},
            {'$set': patient_data},
            upsert=True
        )

        return jsonify({'message': 'PDF, image, form data, and result saved successfully'}), 200
    except Exception as e:
        print(f"Error saving data: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
