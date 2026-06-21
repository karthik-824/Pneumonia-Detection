#  Pneumonia Classifier 

A high-performance deep learning web application designed to assist in the automated analysis of chest CT-Scan images for the detection of pneumonia. Built with **TensorFlow/Keras** and deployed via **Docker** on **Hugging Faces**.

[Try the Live App on Hugging Face](https://huggingface.co/spaces/Karthik-24/Pneumonia-Classifier)

##  Preview
<img width="1000" height="700" alt="image" src="https://github.com/user-attachments/assets/ce377c40-cd57-44cd-93de-d1de09dc83c4" />
<img width="1000" height="700" alt="image" src="https://github.com/user-attachments/assets/5f0f9c8e-5582-4bbc-a7f5-9eae9d708b63" />
<img width="1000" height="700" alt="image" src="https://github.com/user-attachments/assets/5a509e16-0018-4bf5-b35e-b425e55daaea" />

##  Project Overview & Features

This application utilizes a custom-built **MobileNetV2** architecture to classify chest CT-Scan images. The model is optimized for high accuracy while maintaining a lightweight footprint, ensuring efficient inference.

* **Deep Learning Model:** Built using transfer learning (MobileNetV2) and custom top layers.
* **High Accuracy:** The model achieved an accuracy of **~96.7%** with high precision and recall.
* **Web Interface:** A clean, user-friendly frontend built with HTML/CSS that allows users to upload images and instantly get a diagnosis.
* **Cloud Ready:** Fully containerized using Docker and served with Gunicorn for scalable production deployment.

##  Tech Stack

* **Deep Learning:** TensorFlow, Keras, MobileNetV2
* **Backend:** Flask (Python), Gunicorn
* **Data Processing:** OpenCV, NumPy, Pillow
* **Frontend:** HTML5, CSS3, JavaScript (Fetch API)
* **Infrastructure:** Docker, Hugging Face Spaces

##  Methodology

1. **Preprocessing:** Images are normalized to a `256x256` resolution and pixel-scaled to `[0, 1]`.
2. **Architecture:** The model leverages transfer learning principles with a custom top architecture, including `GlobalAveragePooling2D` and `Dropout` layers to prevent overfitting.
3. **Deployment:** The application is containerized using Docker to ensure environment consistency and deployed on a scalable cloud instance.

##  Project Structure

```text
├── app.py                                  # Main Flask application
├── templates
   └── index.html                          # Frontend user interface
├── static
   └── css
   └── js
├── requirements.txt                        # Python dependencies
├── Pneumonia_Classifie.ipynb               # Jupyter Notebook with complete model training code
├── final_trained_model.h5                  # Trained model weights 
└── README.md                               # Project documentation
```

##  How to Run Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/karthik-824/Pneumonia-Detection.git](https://github.com/karthik-824/Pneumonia-Detection.git)
cd Pneumonia-Detection```

**2. Create a virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

**4. Launch the application:**
```bash
python app.py
```
*The web interface will be accessible at `http://localhost:5000` or `http://127.0.0.1:5000`.*

##  Disclaimer
This project is for **educational and research purposes only** and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
