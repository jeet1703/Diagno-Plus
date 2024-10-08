# Diagno Plus+

## Overview

**Diagno Plus+** is an advanced AI-powered medical diagnostic platform that helps users detect various conditions, including skin cancer, breast cancer, brain tumors, and pneumonia. The platform provides accurate and efficient predictions based on medical images, offering early diagnosis to improve patient outcomes. The project is built with a **React** frontend and a **Flask** backend, integrating AI models for image classification.

## Features

- **Skin Cancer Detection**: Predicts if a skin lesion is benign or malignant using a pre-trained ViT model.
- **Breast Cancer Detection**: Classifies breast images into three categories: normal, benign, or malignant.
- **Brain Tumor Detection**: Identifies the type of brain tumor (glioma, meningioma, pituitary tumor, or no tumor).
- **Pneumonia Detection**: Analyzes chest X-rays to detect signs of pneumonia.
- **Chatbot**: Provides answers to medical-related queries using a generative AI model.
- **PDF Report Retrieval**: Allows patients to download their medical reports in PDF format from MongoDB GridFS.
  
## Tech Stack

### Frontend
- **React**: For building the user interface.
- **Tailwind CSS**: For styling the components.
- **React Router**: For navigation between different pages.

### Backend
- **Flask**: As the backend framework to handle API requests.
- **PyTorch**: For running ViT-based cancer prediction models.
- **Keras/TensorFlow**: For brain tumor and pneumonia prediction models.
- **MongoDB**: For storing patient data and reports.
- **GridFS**: To store and retrieve large files (PDFs, images).

### AI Models
- **Vision Transformer (ViT)**: Pre-trained for skin and breast cancer prediction.
- **Keras Models**: For brain tumor and pneumonia predictions.

## Installation

### Prerequisites
- **Python 3.8+**
- **Node.js** and **npm**
- **MongoDB Atlas** account
- **Flask** and necessary Python libraries

### Backend Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo.git
    cd your-repo/backend
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Setup MongoDB connection:
   In the `app.py`, replace the MongoDB URI with your own MongoDB Atlas connection string.

4. Download pre-trained models for cancer and tumor predictions and place them in the appropriate directory.

5. Run the Flask backend:
    ```bash
    python app.py
    ```

### Frontend Setup
1. Navigate to the `frontend` folder:
    ```bash
    cd ../frontend
    ```

2. Install frontend dependencies:
    ```bash
    npm install
    ```

3. Run the React app:
    ```bash
    npm start
    ```

## API Endpoints

- **/api/breastcancer**: POST an image to get breast cancer prediction.
    ```bash
    curl -X POST -F 'file=@path_to_image.jpg' http://localhost:5000/api/breastcancer
    ```

- **/api/skincancer**: POST an image to get skin cancer prediction.
    ```bash
    curl -X POST -F 'file=@path_to_image.jpg' http://localhost:5000/api/skincancer
    ```

- **/api/tumorprediction**: POST an image to predict brain tumor type.
    ```bash
    curl -X POST -F 'file=@path_to_image.jpg' http://localhost:5000/api/tumorprediction
    ```

- **/api/pneumonia-prediction**: POST an image to predict pneumonia.
    ```bash
    curl -X POST -F 'file=@path_to_image.jpg' http://localhost:5000/api/pneumonia-prediction
    ```

- **/api/get-pdf/:patient_id**: GET a PDF report for a patient.
    ```bash
    curl http://localhost:5000/api/get-pdf/patient123
    ```

- **/api/get-image/:patient_id**: GET an image for a patient.
    ```bash
    curl http://localhost:5000/api/get-image/patient123
    ```

- **/api/chat**: POST a query to the chatbot and receive a response.
    ```bash
    curl -X POST -H 'Content-Type: application/json' -d '{"query": "Tell me about skin cancer"}' http://localhost:5000/api/chat
    ```

- **/api/save-pdf**: POST patient data, image, and PDF report to save them.
    ```bash
    curl -X POST -F 'file=@report.pdf' -F 'image=@scan.jpg' -F 'name=John' -F 'age=45' -F 'patientId=123' -F 'result=benign' http://localhost:5000/api/save-pdf
    ```

## License

This project is licensed under the MIT License.
