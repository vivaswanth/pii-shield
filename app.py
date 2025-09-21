# app.py
import streamlit as st
import tempfile
import os
import re
import json
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
import easyocr
from faker import Faker
import io
import zipfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import threading

# ---------------- Global Setup ----------------
REGEX_PATTERNS = {
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE),
    "aadhaar": re.compile(r"\b[2-9]{1}[0-9]{3}\s*[0-9]{4}\s*[0-9]{4}\b"),
    "date": re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}\s+\w+\s+\d{4})\b"
    ),
    "phone": re.compile(r"\+?91[\-\s]?[6-9]{1}[0-9]{9}\b|\b[6-9]{1}[0-9]{9}\b"),
    "email": re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b"),
    "pincode": re.compile(r"\b\d{6}\b"),
}

LABEL_KEYWORDS = [
    "name", "father", "father's", "mother", "parent", "dob", "birth", "date_of_birth",
    "address", "resident", "phone", "mobile", "email", "son", "daughter",
    "s/o", "d/o", "w/o", "to", "from"
]

reader = easyocr.Reader(['en'], gpu=False)
faker = Faker('en_IN')

# ---------------- Utilities ----------------
def create_zip_from_images(images, ext=".jpg"):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
        for idx, img in enumerate(images):
            _, img_bytes = cv2.imencode(ext, img)
            zf.writestr(f"page_{idx+1}{ext}", img_bytes.tobytes())
    zip_buffer.seek(0)
    return zip_buffer.read()

def redact_faces(image):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (51, 51), 30)
        img[y:y+h, x:x+w] = roi
    return img

def redact_image(img, ocr_data, conf_thresh, mode):
    img = img.copy()
    redacted_fields = []
    img = redact_faces(img)

    for i, text in enumerate(ocr_data["text"]):
        t = text.strip()
        if not t:
            continue
        try:
            conf = float(ocr_data["conf"][i])
        except Exception:
            conf = -1
        if conf < conf_thresh:
            continue

        redact = False
        if mode.startswith("Aggressive"):
            redact = True
        else:
            for pattern in REGEX_PATTERNS.values():
                if pattern.search(t):
                    redact = True
                    break
            if not redact:
                for kw in LABEL_KEYWORDS:
                    if kw in t.lower():
                        redact = True
                        break

        if redact:
            x, y, w, h = (
                int(ocr_data["left"][i]),
                int(ocr_data["top"][i]),
                int(ocr_data["width"][i]),
                int(ocr_data["height"][i]),
            )
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
            redacted_fields.append(t)

    return img, redacted_fields

def draw_debug_boxes(img, ocr_data, conf_threshold):
    debug = img.copy()
    for i, txt in enumerate(ocr_data["text"]):
        if not txt.strip():
            continue
        try:
            conf = float(ocr_data["conf"][i])
        except Exception:
            conf = -1
        x, y, w, h = (
            int(ocr_data["left"][i]),
            int(ocr_data["top"][i]),
            int(ocr_data["width"][i]),
            int(ocr_data["height"][i]),
        )
        color = (0, 255, 0) if conf >= conf_threshold else (0, 165, 255)
        cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
        cv2.putText(debug, f"{int(conf)}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return debug

def get_ocr_from_pillow(img_pil):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT, lang="eng")
    return img_cv, data

def process_and_redact_image(image_pil, conf_threshold=30, mode="Smart (labels + patterns)"):
    img_cv, ocr_data = get_ocr_from_pillow(image_pil)
    img_redacted, redacted_fields = redact_image(img_cv, ocr_data, conf_threshold, mode)
    return img_cv, img_redacted, redacted_fields, ocr_data

def handle_pdf(uploaded_bytes, conf_threshold=30, mode="Smart (labels + patterns)"):
    pdf_images = convert_from_bytes(uploaded_bytes)
    all_pages_original = []
    all_pages_redacted = []
    all_redacted_fields = []
    for page in pdf_images:
        orig, redacted, redacted_fields, _ = process_and_redact_image(page, conf_threshold, mode)
        all_pages_original.append(orig)
        all_pages_redacted.append(redacted)
        all_redacted_fields.extend(redacted_fields)
    return all_pages_original, all_pages_redacted, all_redacted_fields

def apply_faker_rule(rule):
    ftype = rule.get("faker", "mask")
    if ftype == "mask":
        return "***REDACTED***"
    elif ftype == "name":
        return faker.name()
    elif ftype == "name_with_title":
        return f"{faker.prefix()} {faker.name()}"
    elif ftype == "email":
        return faker.email()
    elif ftype == "phone_number":
        return faker.phone_number()
    elif ftype == "date":
        dt = faker.date_of_birth(minimum_age=18, maximum_age=90)
        fmt = rule.get("format", "%d-%m-%Y")
        return dt.strftime(fmt)
    elif ftype == "ssn":
        return faker.ssn()
    elif ftype == "bban":
        return faker.bban()
    elif ftype == "random_number":
        return faker.random_int(min=rule.get("min", 0), max=rule.get("max", 999999))
    elif ftype == "address":
        return faker.address()
    else:
        return "***REDACTED***"

def redact_json_with_faker_config(data, config):
    redacted_fields = []

    def _redact(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                matched_rule = None
                for cfg_key, cfg_rule in config.items():
                    if k.lower() == cfg_key.lower():
                        matched_rule = cfg_rule
                        break
                if matched_rule:
                    out[k] = apply_faker_rule(matched_rule)
                    redacted_fields.append(k)
                else:
                    out[k] = _redact(v)
            return out
        elif isinstance(obj, list):
            return [_redact(v) for v in obj]
        else:
            return obj

    redacted_data = _redact(data)
    return redacted_data, redacted_fields

# ---------------- FastAPI Backend ----------------
app = FastAPI(title="PII Shield API")

@app.post("/redact/image")
async def api_redact_image(file: UploadFile = File(...), conf_threshold: int = Form(30), mode: str = Form("Smart (labels + patterns)")):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    _, redacted_cv, redacted_fields, _ = process_and_redact_image(image, conf_threshold, mode)
    out_bytes = cv2.imencode('.jpg', redacted_cv)[1].tobytes()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp_file.write(out_bytes)
    tmp_file.close()
    return FileResponse(tmp_file.name, media_type="image/jpeg", filename="redacted.jpg")

@app.post("/redact/json")
async def api_redact_json(file: UploadFile = File(...), config_file: UploadFile = File(None)):
    data = json.load(file.file)
    faker_config = json.load(config_file.file) if config_file else {}
    redacted_json, redacted_fields = redact_json_with_faker_config(data, faker_config)
    return JSONResponse(content={"redacted_json": redacted_json, "redacted_fields": redacted_fields})

# ---------------- Streamlit UI ----------------
def run_streamlit_ui():
    st.set_page_config(page_title="PII Shield", page_icon="🛡️", layout="centered")
    st.header("🛡️ PII Shield")
    st.markdown(
    "<div style='text-align:left;font-size: 1rem;'>"
    "Upload an image, PDF, or JSON file containing PII.<br/>"
    "Select a suitable redaction mode and threshold in settings tab. Download your safe, redacted version."
    "</div>",
    unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Sidebar Tabs
    st.sidebar.title("PII Shield Options")
    tab = st.sidebar.radio("Select tab", ["Image/PDF", "JSON", "Settings"])
    
    if tab == "Settings":
        st.header("Settings")
        conf_threshold = st.slider("OCR confidence threshold", 0, 100, 30)
        mode = st.radio("Redaction mode", ["Smart (labels + patterns)", "Aggressive (all text blocks)"])
        show_debug = st.checkbox("Show debug boxes/labels (dev)", False)
        st.session_state.update({"conf_threshold": conf_threshold, "mode": mode, "show_debug": show_debug})
    
    else:
        conf_threshold = st.session_state.get("conf_threshold", 30)
        mode = st.session_state.get("mode", "Smart (labels + patterns)")
        show_debug = st.session_state.get("show_debug", False)
        
        uploaded_file = st.file_uploader("Upload file", type=["jpg","jpeg","png","pdf","json"])
        config_file = st.file_uploader("Optional: Faker Config JSON", type=["json"])
        faker_config = json.load(config_file) if config_file else {}
        
        if uploaded_file:
            name, ext = os.path.splitext(uploaded_file.name.lower())
            if ext in [".jpg", ".jpeg", ".png"]:
                image = Image.open(uploaded_file).convert("RGB")
                orig_cv, redacted_cv, redacted_fields, ocr_data = process_and_redact_image(image, conf_threshold, mode)
                st.subheader("Original Image")
                st.image(cv2.cvtColor(orig_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.subheader("Redacted Image")
                st.image(cv2.cvtColor(redacted_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.download_button("Download Redacted Image", cv2.imencode('.jpg', redacted_cv)[1].tobytes(), "redacted.jpg")
                if show_debug:
                    st.subheader("Debug OCR")
                    st.image(cv2.cvtColor(draw_debug_boxes(orig_cv, ocr_data, conf_threshold), cv2.COLOR_BGR2RGB))
            
            elif ext == ".pdf":
                orig_imgs, redact_imgs, redacted_fields = handle_pdf(uploaded_file.read(), conf_threshold, mode)
                for idx, (orig, redacted) in enumerate(zip(orig_imgs, redact_imgs)):
                    st.subheader(f"Page {idx+1} Original")
                    st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.subheader(f"Page {idx+1} Redacted")
                    st.image(cv2.cvtColor(redacted, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.download_button("Download ZIP", create_zip_from_images(redact_imgs), "redacted_pages.zip")
            
            elif ext == ".json":
                data = json.load(uploaded_file)
                redacted_json, redacted_fields = redact_json_with_faker_config(data, faker_config)
                st.subheader("Original JSON")
                st.json(data, expanded=True)
                st.subheader("Redacted JSON")
                st.json(redacted_json, expanded=True)
                st.download_button("Download Redacted JSON", json.dumps(redacted_json, indent=2), "redacted.json")
    
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9em;'>"
        "PII Shield &copy; 2025. Open Source. <br>Secure, offline, no data stored."
        "</div>", unsafe_allow_html=True
    )

# ---------------- Run ----------------
if __name__ == "__main__":
    # Run FastAPI in a separate thread
    # threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info"), daemon=True).start()
    # Run Streamlit UI
    run_streamlit_ui()
