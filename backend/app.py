import time
import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from utils import (
    extract_text_from_file,
    predict_job_offer,
    predict_email,
    predict_sms_or_url,
    predict_deepfake_image,
    generate_forensic_pdf,
    evaluate_adversarial_robustness
)

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "active",
        "system": "AI ScamShield Multi-Modal Framework",
        "version": "1.4.0-enterprise"
    })

@app.route('/api/v1/sandbox/execute', methods=['POST'])
def api_sandbox_execute():
    start_time = time.time()
    data = request.get_json() or {}
    endpoint_type = data.get('module', 'job')
    payload = data.get('payload', '')

    if endpoint_type == 'job':
        result = predict_job_offer(payload)
    elif endpoint_type == 'email':
        result = predict_email(payload)
    elif endpoint_type == 'sms_url':
        result = predict_sms_or_url(payload)
    else:
        result = predict_job_offer(payload)

    latency_ms = round((time.time() - start_time) * 1000, 2)

    response_envelope = {
        "status_code": 200,
        "api_version": "v1.4.0-enterprise",
        "execution_latency_ms": latency_ms,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "engine_telemetry": {
            "node": "SECURE-NODE-US-EAST-01",
            "encryption": "AES-256-GCM"
        },
        "data": result
    }
    return jsonify(response_envelope)

@app.route('/api/adversarial-test', methods=['POST'])
def api_adversarial_test():
    if 'file' in request.files:
        file = request.files['file']
        result = evaluate_adversarial_robustness('image', file.read())
    else:
        data = request.get_json() or {}
        payload = data.get('payload', 'Global Logistics Partners LLC remote data entry offer')
        result = evaluate_adversarial_robustness('text', payload)
    return jsonify(result)

@app.route('/api/predict/job', methods=['POST'])
def api_predict_job():
    if 'file' in request.files:
        file = request.files['file']
        text = extract_text_from_file(file.read(), file.filename)
    else:
        data = request.get_json() or {}
        text = data.get('text', '')
        
    if not text:
        return jsonify({"error": "No text content or document file provided."}), 400
        
    return jsonify(predict_job_offer(text))

@app.route('/api/predict/email', methods=['POST'])
def api_predict_email():
    if 'file' in request.files:
        file = request.files['file']
        text = extract_text_from_file(file.read(), file.filename)
    else:
        data = request.get_json() or {}
        text = data.get('text', '')
        
    if not text:
        return jsonify({"error": "No email body or document provided."}), 400
        
    return jsonify(predict_email(text))

@app.route('/api/predict/sms_url', methods=['POST'])
def handle_sms_url_prediction():
    if 'file' in request.files:
        file = request.files['file']
        text = extract_text_from_file(file.read(), file.filename)
    else:
        data = request.get_json() or {}
        text = data.get('text', '')
        
    if not text:
        return jsonify({"error": "Missing 'text' field or file content"}), 400
    
    result = predict_sms_or_url(text)
    return jsonify(result)

@app.route('/api/predict/image', methods=['POST'])
def handle_image_prediction():
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['file']
    image_bytes = file.read()
    
    result = predict_deepfake_image(image_bytes)
    return jsonify(result)

@app.route('/api/download-report', methods=['POST'])
def download_report():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing report metadata"}), 400
    
    pdf_buffer = generate_forensic_pdf(data)
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"Forensic_Report_{data.get('module', 'Scan').replace(' ', '_')}.pdf"
    )



if __name__ == '__main__':
    print("🚀 Starting AI ScamShield Backend Engine with CORS...")
    app.run(host='0.0.0.0', port=5000, debug=True)