# AI-ScamShield
AI ScamShield is a cybersecurity microservice built with Flask, TensorFlow &amp; Scikit-Learn to detect phishing, fake contracts &amp; deepfakes. It uses Explainable AI (XAI) for visual proof (heatmaps &amp; token weights) and ensures complete data privacy via zero-retention memory, generating SHA-256 PDF audit certificates.
# AI ScamShield 🛡️
**An Enterprise-Grade Multi-Modal Threat Intelligence & Digital Forensic Platform**

AI ScamShield is a multi-modal cybersecurity microservice designed to detect digital fraud, including fake job contracts, phishing emails, malicious URLs, and AI-generated deepfakes. Developed as an M.Sc. Computer Science dissertation project, it utilizes **Explainable AI (XAI)** to transparently prove its reasoning and guarantees user privacy through a strict zero-data-retention architecture.

## 🚀 Key Features
* **Multi-Modal Ingestion:** Simultaneously processes raw text, URLs, PDF/Word documents, and images.
* **NLP & URL Threat Engine:** Detects structural extortion, advance-fee fraud, and typosquatting using Sublinear TF-IDF and Logistic Regression.
* **Vision Forensics:** Identifies AI-generated deepfakes and manipulated synthetic media using a MobileNetV2 Convolutional Neural Network (CNN) and Fast Fourier Transform (FFT).
* **Explainable AI (XAI):** Rejects the "black box" AI approach by providing Token/Feature Attribution for text and visual Grad-CAM thermal heatmaps for manipulated image pixels.
* **Zero-Retention Privacy:** Processes all sensitive files in temporary volatile memory (`io.BytesIO()`) ensuring no user data is ever written to disk.
* **Cryptographic Reporting:** Automatically generates tamper-evident, SHA-256 cryptographically signed PDF audit certificates via ReportLab.

## 💻 Tech Stack
* **Frontend:** HTML5, JavaScript (ES6), Tailwind CSS (Dark Mode/Obsidian UI)
* **Backend Gateway:** Python 3.10, Flask, Flask-CORS, PythonAnywhere PaaS
* **Machine Learning:** Scikit-Learn, TensorFlow / Keras
* **Computer Vision:** OpenCV, Pillow (PIL)
* **Data Extraction:** Tesseract OCR (`pytesseract`), `pdfplumber`, `python-docx`

## ⚙️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/AI-ScamShield.git](https://github.com/YOUR_USERNAME/AI-ScamShield.git)
   cd AI-ScamShield

   Create a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
Run the application:

Bash
flask run
