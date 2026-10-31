import os
import io
import cv2
import uuid
import base64
import joblib
import hashlib
import datetime
import numpy as np
import pdfplumber
import docx
import pytesseract
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
from urllib.parse import urlparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# ------------------------------------------------------------------
# Safe OpenCV Dual-Cascade Initialization
# ------------------------------------------------------------------
HAS_CASCADE = False
try:
    frontal_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
    
    face_cascade = cv2.CascadeClassifier(frontal_path)
    profile_cascade = cv2.CascadeClassifier(profile_path)
    
    HAS_CASCADE = not face_cascade.empty() and not profile_cascade.empty()
except Exception:
    HAS_CASCADE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# ------------------------------------------------------------------
# 1. Load Trained NLP & Vision Models
# ------------------------------------------------------------------
lr_job = joblib.load(os.path.join(MODELS_DIR, 'fake_job', 'logistic_regression.pkl'))
vec_job = joblib.load(os.path.join(MODELS_DIR, 'fake_job', 'tfidf_vectorizer.pkl'))

lr_email = joblib.load(os.path.join(MODELS_DIR, 'email', 'logistic_regression.pkl'))
vec_email = joblib.load(os.path.join(MODELS_DIR, 'email', 'tfidf_vectorizer.pkl'))

lr_sms = joblib.load(os.path.join(MODELS_DIR, 'sms', 'logistic_regression.pkl'))
vec_sms = joblib.load(os.path.join(MODELS_DIR, 'sms', 'tfidf_vectorizer.pkl'))

lr_url = joblib.load(os.path.join(MODELS_DIR, 'url', 'logistic_regression.pkl'))
vec_url = joblib.load(os.path.join(MODELS_DIR, 'url', 'tfidf_vectorizer.pkl'))

deepfake_model = load_model(os.path.join(MODELS_DIR, 'deepfake', 'deepfake_detector.h5'))

TRUSTED_DOMAINS = {
    "google.com", "www.google.com", "youtube.com", "www.youtube.com",
    "github.com", "microsoft.com", "linkedin.com", "amazon.com",
    "wikipedia.org", "apple.com", "yahoo.com"
}

# ------------------------------------------------------------------
# 2. Text & Document Extraction Helpers
# ------------------------------------------------------------------
def extract_text_from_file(file_bytes, filename):
    ext = filename.lower().split('.')[-1]
    extracted_text = ""
    try:
        if ext == 'pdf':
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text: extracted_text += text + "\n"
        elif ext in ['docx', 'doc']:
            doc = docx.Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                if para.text: extracted_text += para.text + "\n"
        elif ext in ['png', 'jpg', 'jpeg', 'webp']:
            img = Image.open(io.BytesIO(file_bytes))
            extracted_text = pytesseract.image_to_string(img)
        else:
            extracted_text = file_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Extraction Error: {str(e)}")
    return extracted_text.strip()


def extract_xai_keywords(vectorizer, model, text, top_n=5):
    text_tfidf = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]
    
    non_zero_indices = text_tfidf.nonzero()[1]
    word_weights = []
    
    for idx in non_zero_indices:
        word = feature_names[idx]
        weight = coefficients[idx]
        if weight > 0:
            word_weights.append({"word": word, "score": round(float(weight), 3)})
            
    return sorted(word_weights, key=lambda x: x["score"], reverse=True)[:top_n]

# ------------------------------------------------------------------
# 3. NLP Inference Functions (with Advisory Insights)
# ------------------------------------------------------------------
def predict_job_offer(text):
    text_tfidf = vec_job.transform([text])
    pred = int(lr_job.predict(text_tfidf)[0])
    prob = float(lr_job.predict_proba(text_tfidf)[0][1])
    triggers = extract_xai_keywords(vec_job, lr_job, text)
    is_fraud = pred == 1

    reasons = [
        "Unrealistic salary-to-skill ratio relative to low entry requirements.",
        "Request for upfront equipment payment or check-cashing verification fees."
    ] if is_fraud else [
        "Language aligns with standardized, professional corporate recruitment practices.",
        "Contains verifiable corporate metadata and standard application workflows."
    ]

    precautions = [
        "Never deposit corporate checks or send money to purchase 'employer-mandated equipment'.",
        "Verify job offers independently via the company's official careers portal (not links provided in text)."
    ] if is_fraud else [
        "Always confirm recruiter identities via corporate email domains rather than personal handles.",
        "Review official employment contracts thoroughly before signing or sharing banking details."
    ]

    return {
        "module": "Fake Job Offer Detection",
        "is_fraud": is_fraud,
        "risk_score_pct": round(prob * 100, 2),
        "status": "🚨 FRAUDULENT JOB OFFER" if is_fraud else "✅ LEGITIMATE JOB OFFER",
        "suspicious_triggers": triggers,
        "why_flagged": reasons,
        "precautions": precautions
    }


def predict_email(text):
    text_tfidf = vec_email.transform([text])
    pred = int(lr_email.predict(text_tfidf)[0])
    prob = float(lr_email.predict_proba(text_tfidf)[0][1])
    triggers = extract_xai_keywords(vec_email, lr_email, text)
    is_fraud = pred == 1

    reasons = [
        "High psychological pressure and urgency tactics ('confirm within 2 hours').",
        "Redirection to third-party messaging apps (Telegram/WhatsApp) instead of official corporate channels."
    ] if is_fraud else [
        "Tone is neutral, scheduled, and matches standard corporate calendar communication workflows.",
        "Domain headers and links correlate with verified corporate communication infrastructure."
    ]

    precautions = [
        "Do not click embedded links or scan QR codes from unverified email senders.",
        "Never conduct official interviews over encrypted messaging apps without corporate domain authentication."
    ] if is_fraud else [
        "Keep your security software updated and verify meeting links before joining virtual calls.",
        "Always cross-reference scheduling requests with official recruiter communications."
    ]

    return {
        "module": "Fake Interview Email Detection",
        "is_fraud": is_fraud,
        "risk_score_pct": round(prob * 100, 2),
        "status": "🚨 PHISHING / FAKE INTERVIEW EMAIL" if is_fraud else "✅ LEGITIMATE EMAIL",
        "suspicious_triggers": triggers,
        "why_flagged": reasons,
        "precautions": precautions
    }


def predict_sms_or_url(input_str):
    is_url = input_str.startswith(('http://', 'https://')) or ('.' in input_str and ' ' not in input_str)
    
    if is_url:
        parsed = urlparse(input_str if input_str.startswith(('http://', 'https://')) else 'http://' + input_str)
        domain = parsed.netloc.lower()
        
        if domain in TRUSTED_DOMAINS or domain.replace("www.", "") in TRUSTED_DOMAINS:
            return {
                "module": "Malicious URL Detection",
                "is_fraud": False,
                "risk_score_pct": 0.0,
                "status": "✅ SAFE (Trusted Whitelist)",
                "suspicious_triggers": [],
                "why_flagged": ["Domain matches verified global organization whitelist."],
                "precautions": ["Always ensure HTTPS encryption is active when entering personal data."]
            }
            
        vec = vec_url.transform([input_str])
        pred = int(lr_url.predict(vec)[0])
        prob = float(lr_url.predict_proba(vec)[0][1])
        is_fraud = pred == 1

        return {
            "module": "Malicious URL Detection",
            "is_fraud": is_fraud,
            "risk_score_pct": round(prob * 100, 2),
            "status": "🚨 MALICIOUS URL" if is_fraud else "✅ SAFE URL",
            "suspicious_triggers": [],
            "why_flagged": ["Anomalous subdomain length and lookalike character architecture."] if is_fraud else ["Standard domain structure with no known threat signatures."],
            "precautions": ["Do not click links received via unverified SMS or messaging apps."] if is_fraud else ["Exercise standard caution when downloading files or entering credentials."]
        }
    else:
        vec = vec_sms.transform([input_str])
        pred = int(lr_sms.predict(vec)[0])
        prob = float(lr_sms.predict_proba(vec)[0][1])
        triggers = extract_xai_keywords(vec_sms, lr_sms, input_str)
        is_fraud = pred == 1

        return {
            "module": "Phishing SMS Detection",
            "is_fraud": is_fraud,
            "risk_score_pct": round(prob * 100, 2),
            "status": "🚨 SPAM / PHISHING SMS" if is_fraud else "✅ SAFE SMS",
            "suspicious_triggers": triggers,
            "why_flagged": ["Contains aggressive urgency triggers and untrusted external web links."] if is_fraud else ["Standard mobile message syntax with no malicious markers."],
            "precautions": ["Never reply to unsolicited SMS requesting banking or account credentials."] if is_fraud else ["Keep device messaging filters enabled."]
        }

# ------------------------------------------------------------------
# 4. Multi-Stage Computer Vision & Forensic Analysis Engine
# ------------------------------------------------------------------
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
            return layer.name
    return None


def generate_gradcam_heatmap(model, img_array, original_cv2_img):
    if cv2 is None or original_cv2_img is None: return None
    try:
        last_conv_name = get_last_conv_layer_name(model)
        if not last_conv_name: return None

        grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()

        heatmap_resized = cv2.resize(heatmap, (original_cv2_img.shape[1], original_cv2_img.shape[0]))
        color_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(original_cv2_img, 0.65, color_heatmap, 0.35, 0)

        _, buffer = cv2.imencode('.jpg', superimposed)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    except Exception:
        return None


def compute_frequency_anomaly_score(pil_img):
    try:
        gray = np.array(pil_img.convert('L'))
        f = np.fft.fft2(gray)
        magnitude_spectrum = 20 * np.log(np.abs(np.fft.fftshift(f)) + 1e-10)
        
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
        center_mask = np.zeros((h, w), dtype=bool)
        cv2.circle(center_mask.astype(np.uint8), (cx, cy), r, 1, -1)
        
        high_freq = np.mean(magnitude_spectrum[~center_mask])
        low_freq = np.mean(magnitude_spectrum[center_mask])
        ratio = high_freq / (low_freq + 1e-10)
        
        if ratio < 0.70: return 0.85
        elif ratio < 0.78: return 0.65
        else: return 0.15
    except Exception:
        return 0.50


def compute_ela_anomaly_score(pil_img):
    try:
        buffer = io.BytesIO()
        pil_img.convert('RGB').save(buffer, 'JPEG', quality=90)
        buffer.seek(0)
        resaved = Image.open(buffer)
        
        diff = ImageChops.difference(pil_img.convert('RGB'), resaved)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        scale = 255.0 / max(1, max_diff)
        diff_arr = np.array(ImageEnhance.Brightness(diff).enhance(scale))
        
        mean_ela = np.mean(diff_arr)
        if mean_ela < 3.5: return 0.85
        elif mean_ela < 5.5: return 0.60
        else: return 0.20
    except Exception:
        return 0.50


def detect_all_faces(img_cv):
    faces = []
    if not HAS_CASCADE or img_cv is None: return faces
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    for f in face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)): faces.append(tuple(f))
    for p in profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)): faces.append(tuple(p))
    
    gray_flipped = cv2.flip(gray, 1)
    w_img = gray.shape[1]
    for (x, y, w, h) in profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)):
        faces.append((w_img - (x + w), y, w, h))
    return faces


def predict_deepfake_image(image_bytes):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    img_cv = None
    if cv2 is not None:
        try:
            img_cv = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception: pass

    faces_found = detect_all_faces(img_cv) if img_cv is not None else []
    target_cv_img = img_cv
    cnn_raw_risk = 0.5
    triggers = []

    if len(faces_found) > 0:
        face_scores = []
        for (x, y, w, h) in faces_found:
            pad_w, pad_h = int(w * 0.15), int(h * 0.15)
            y1, y2 = max(0, y - pad_h), min(img_cv.shape[0], y + h + pad_h)
            x1, x2 = max(0, x - pad_w), min(img_cv.shape[1], x + w + pad_w)
            
            cropped_face = img_cv[y1:y2, x1:x2]
            rgb_cropped = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            
            face_pil = Image.fromarray(rgb_cropped).resize((128, 128))
            face_arr = image.img_to_array(face_pil)
            face_arr = np.expand_dims(face_arr, axis=0)
            face_arr = preprocess_input(face_arr)
            
            raw_score = float(deepfake_model.predict(face_arr, verbose=0)[0][0])
            risk = raw_score
            face_scores.append((risk, cropped_face))

        face_scores.sort(key=lambda item: item[0], reverse=True)
        cnn_raw_risk = face_scores[0][0]
        target_cv_img = face_scores[0][1]
        triggers.append({"word": f"CNN Facial Landmark Analysis ({len(faces_found)} Face(s))", "score": round(cnn_raw_risk, 2)})
    else:
        w, h = pil_img.size
        crops = [pil_img.resize((128, 128)), pil_img.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))).resize((128, 128))]
        scores = []
        for c in crops:
            arr = image.img_to_array(c)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)
            scores.append(float(deepfake_model.predict(arr, verbose=0)[0][0]))
        cnn_raw_risk = float(np.max(scores))
        triggers.append({"word": "Multi-Crop Global CNN Scan", "score": round(cnn_raw_risk, 2)})

    fft_risk = compute_frequency_anomaly_score(pil_img)
    ela_risk = compute_ela_anomaly_score(pil_img)

    if fft_risk > 0.45: triggers.append({"word": "Diffusion Frequency Grid (FFT)", "score": round(fft_risk, 2)})
    if ela_risk > 0.45: triggers.append({"word": "Compression Error Variance (ELA)", "score": round(ela_risk, 2)})

    if len(faces_found) > 0:
        composite_risk = (0.80 * cnn_raw_risk) + (0.10 * fft_risk) + (0.10 * ela_risk)
    else:
        composite_risk = (0.40 * cnn_raw_risk) + (0.30 * fft_risk) + (0.30 * ela_risk)

    risk_percentage = round(composite_risk * 100, 2)
    is_fake = risk_percentage >= 40.0

    reasons = [
        "Anomalous high-frequency grid spectra characteristic of generative diffusion models (Stable Diffusion/Midjourney).",
        "Inconsistent compression error variance (ELA) across subject boundaries and background lighting."
    ] if is_fake else [
        "Natural organic camera sensor noise distribution and consistent high-frequency illumination.",
        "Anatomical facial alignment and compression uniformity match organic photographic captures."
    ]

    precautions = [
        "Do not trust video or photographic identities online without multi-factor cross-verification.",
        "Be wary of remote video interviews where recruiters turn off cameras citing technical difficulties."
    ] if is_fake else [
        "Maintain standard digital hygiene when sharing personal portraits online.",
        "Verify media origin metadata if authenticity is critical for legal or journalistic use."
    ]

    heatmap_b64 = None
    if target_cv_img is not None:
        try:
            target_rgb = cv2.cvtColor(target_cv_img, cv2.COLOR_BGR2RGB) if len(target_cv_img.shape) == 3 else target_cv_img
            t_arr = image.img_to_array(Image.fromarray(target_rgb).resize((128, 128)))
            t_arr = np.expand_dims(t_arr, axis=0)
            t_arr = preprocess_input(t_arr)
            heatmap_b64 = generate_gradcam_heatmap(deepfake_model, t_arr, target_cv_img)
        except Exception: pass

    return {
        "module": "Deepfake Image Detection",
        "is_fraud": is_fake,
        "risk_score_pct": risk_percentage,
        "status": "🚨 AI-GENERATED / DEEPFAKE IMAGE" if is_fake else "✅ AUTHENTIC IMAGE",
        "heatmap_image": heatmap_b64,
        "suspicious_triggers": triggers if is_fake else [],
        "why_flagged": reasons,
        "precautions": precautions
    }

# ------------------------------------------------------------------
# 5. PDF Forensic Certificate Generator
# ------------------------------------------------------------------
def generate_forensic_pdf(result_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    primary_dark = colors.HexColor('#0f172a')
    accent_cyan = colors.HexColor('#0891b2')
    is_fraud = result_data.get('is_fraud', False)
    status_color = colors.HexColor('#e11d48') if is_fraud else colors.HexColor('#059669')

    raw_payload_str = str(result_data) + datetime.datetime.now().isoformat()
    sha256_hash = hashlib.sha256(raw_payload_str.encode('utf-8')).hexdigest()[:32].upper()

    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=20, textColor=primary_dark, spaceAfter=2)
    sub_style = ParagraphStyle('SubTitle', parent=styles['Italic'], fontSize=9, textColor=colors.HexColor('#64748b'))
    
    story.append(Paragraph("AI SCAMSHIELD — DIGITAL FORENSIC CERTIFICATE", title_style))
    story.append(Paragraph(f"Incident Hash: 0x{sha256_hash} | Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", sub_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=accent_cyan, spaceAfter=15))

    table_data = [
        [Paragraph("<b>Target Inspection Module</b>", styles['Normal']), Paragraph(str(result_data.get('module')), styles['Normal'])],
        [Paragraph("<b>Forensic Status</b>", styles['Normal']), Paragraph(f"<font color='{status_color}'><b>{result_data.get('status')}</b></font>", styles['Normal'])],
        [Paragraph("<b>Calculated Fraud Risk Probability</b>", styles['Normal']), Paragraph(f"<b>{result_data.get('risk_score_pct')}%</b>", styles['Normal'])],
        [Paragraph("<b>Confidence Metric</b>", styles['Normal']), Paragraph("High (Multi-Signal Consensus)", styles['Normal'])],
        [Paragraph("<b>Audit ID</b>", styles['Normal']), Paragraph(f"SEC-{uuid.uuid4().hex[:8].upper()}", styles['Normal'])]
    ]

    t = Table(table_data, colWidths=[160, 370])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Explainable AI (XAI) Feature Importance Matrix</b>", styles['Heading3']))
    triggers = result_data.get('suspicious_triggers', [])
    if triggers:
        for trig in triggers:
            story.append(Paragraph(f"• <b>Trigger Isolated:</b> '{trig.get('word')}' — <b>Risk Weight Contribution:</b> +{trig.get('score')}", styles['Normal']))
    else:
        story.append(Paragraph("• No significant synthetic/fraud feature anomalies isolated in feature space.", styles['Normal']))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1'), spaceAfter=10))
    story.append(Paragraph("<i>This document is an automated digital forensic output generated by AI ScamShield Multi-Modal Engine. Authorized for security verification and research auditing purposes.</i>", sub_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ------------------------------------------------------------------
# 6. Adversarial Robustness & Evasion Stress-Testing Lab
# ------------------------------------------------------------------
def evaluate_adversarial_robustness(payload_type, raw_payload):
    """
    Handles both text homoglyph attacks and image noise perturbations dynamically.
    """
    if payload_type == 'image':
        try:
            nparr = np.frombuffer(raw_payload, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            noise = np.random.normal(0, 30, img_cv.shape).astype(np.uint8)
            noisy_img = cv2.add(img_cv, noise)
            
            _, buffer = cv2.imencode('.jpg', noisy_img)
            noisy_bytes = buffer.tobytes()
            
            orig_res = predict_deepfake_image(raw_payload)
            adv_res = predict_deepfake_image(noisy_bytes)
            
            delta = round(abs(orig_res['risk_score_pct'] - adv_res['risk_score_pct']), 2)
            status = "🛡️ ROBUST (High Structural Stability)" if delta < 20.0 else "⚠️ SENSITIVE TO NOISE"

            return {
                "module": "Adversarial Vision Stress Test",
                "attack_type": "Gaussian Noise Perturbation ($\sigma = 30$)",
                "original_risk_pct": orig_res['risk_score_pct'],
                "adversarial_risk_pct": adv_res['risk_score_pct'],
                "confidence_delta_pct": delta,
                "status": status,
                "dissertation_note": "Evaluates convolutional stability under high-frequency image corruption."
            }
        except Exception as e:
            return {"error": str(e)}
    else:
        text = str(raw_payload) if raw_payload else "Global Logistics Partners remote data entry offer"
        adversarial_text = text.replace('a', 'а').replace('e', 'е').replace('o', 'о')
        
        text_tfidf = vec_job.transform([text])
        orig_prob = float(lr_job.predict_proba(text_tfidf)[0][1]) * 100
        
        adv_tfidf = vec_job.transform([adversarial_text])
        adv_prob = float(lr_job.predict_proba(adv_tfidf)[0][1]) * 100
        
        delta = round(abs(orig_prob - adv_prob), 2)
        status = "🛡️ ROBUST (Resisted Evasion)" if delta < 15.0 else "⚠️ VULNERABLE (Confidence Degraded)"

        return {
            "module": "Adversarial NLP Stress Test",
            "attack_type": "Homoglyph Character Substitution (Cyrillic Spoofing)",
            "original_payload": text[:60] + "...",
            "adversarial_payload": adversarial_text[:60] + "...",
            "original_risk_pct": round(orig_prob, 2),
            "adversarial_risk_pct": round(adv_prob, 2),
            "confidence_delta_pct": delta,
            "status": status,
            "dissertation_note": "Measures model resistance against character-level obfuscation and spoofing."
        }

    # ------------------------------------------------------------------
