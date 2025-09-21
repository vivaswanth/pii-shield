# pii_shield.py
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
import requests
from faker import Faker
import io
import zipfile

# ---------------- Streamlit Page ----------------
st.set_page_config(page_title="PII Shield", page_icon="🛡️", layout="centered")
st.markdown("<h1 style='text-align: center;'>🛡️ PII Shield</h1>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;font-size: 1.1rem;'>"
    "Upload an image, PDF, or JSON file containing PII.<br/>"
    "<i>Select a suitable redaction mode and threshold. Download your safe, redacted version.</i>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------- Settings ----------------
mode = st.radio("Redaction mode", ["Smart (labels + patterns)", "Aggressive (all text blocks)"])
conf_threshold = st.slider("OCR confidence threshold", 0, 100, 30)
show_debug = st.checkbox("Show debug boxes/labels (dev)", False)

# ---------------- Regex & Keywords ----------------
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
def section_block(title):
    st.markdown(f"<div style='margin: 2em 0 0.5em 0;'><b>{title}</b></div>", unsafe_allow_html=True)

def create_zip_from_images(images, ext=".jpg"):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
        for idx, img in enumerate(images):
            _, img_bytes = cv2.imencode(ext, img)
            zf.writestr(f"page_{idx+1}{ext}", img_bytes.tobytes())
    zip_buffer.seek(0)
    return zip_buffer.read()

# ---------------- OCR & Image Redaction ----------------
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

def process_and_redact_image(image_pil):
    img_cv, ocr_data = get_ocr_from_pillow(image_pil)
    img_redacted, redacted_fields = redact_image(img_cv, ocr_data, conf_threshold, mode)
    return img_cv, img_redacted, redacted_fields, ocr_data

def handle_pdf(uploaded_bytes):
    pdf_images = convert_from_bytes(uploaded_bytes)
    all_pages_original = []
    all_pages_redacted = []
    all_redacted_fields = []
    for page in pdf_images:
        orig, redacted, redacted_fields, _ = process_and_redact_image(page)
        all_pages_original.append(orig)
        all_pages_redacted.append(redacted)
        all_redacted_fields.extend(redacted_fields)
    return all_pages_original, all_pages_redacted, all_redacted_fields

# ---------------- JSON Redaction with Faker ----------------
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
                # Check if this key matches any config key (case-insensitive)
                matched_rule = None
                for cfg_key, cfg_rule in config.items():
                    if k.lower() == cfg_key.lower():
                        matched_rule = cfg_rule
                        break

                if matched_rule:
                    out[k] = apply_faker_rule(matched_rule)
                    redacted_fields.append(k)
                else:
                    # Recurse into nested dict/list
                    out[k] = _redact(v)
            return out
        elif isinstance(obj, list):
            return [_redact(v) for v in obj]
        else:
            return obj

    redacted_data = _redact(data)
    return redacted_data, redacted_fields



# ---------------- Main Streamlit Flow ----------------
uploaded_file = st.file_uploader(
    "Upload image (PAN/Aadhaar), PDF, or JSON",
    type=["jpg", "jpeg", "png", "pdf", "json"],
    accept_multiple_files=False,
)

config_file = st.file_uploader("Optional: Upload Faker config JSON", type=["json"], key="config")
faker_config = json.load(config_file) if config_file else {}

api_url = st.text_input("Or enter API URL to fetch JSON")

if uploaded_file or api_url:
    with st.spinner("Processing..."):
        redacted_fields_total = []

        # ---------------- Image ----------------
        if uploaded_file:
            name, ext = os.path.splitext(uploaded_file.name.lower())
            if ext in [".jpg", ".jpeg", ".png"]:
                image = Image.open(uploaded_file).convert("RGB")
                orig_cv, redacted_cv, redacted_fields, ocr_data = process_and_redact_image(image)

                tab_orig, tab_red = st.tabs(["Original", "Redacted"])
                with tab_orig:
                    st.image(cv2.cvtColor(orig_cv, cv2.COLOR_BGR2RGB), use_column_width=True)
                with tab_red:
                    st.image(cv2.cvtColor(redacted_cv, cv2.COLOR_BGR2RGB), use_column_width=True)
                    st.download_button(
                        "Download redacted image",
                        data=cv2.imencode('.jpg', redacted_cv)[1].tobytes(),
                        file_name="redacted.jpg",
                        mime="image/jpeg"
                    )
                if show_debug:
                    section_block("Debug OCR")
                    st.image(cv2.cvtColor(draw_debug_boxes(orig_cv, ocr_data, conf_threshold), cv2.COLOR_BGR2RGB))

                redacted_fields_total.extend(redacted_fields)

            # ---------------- PDF ----------------
            elif ext == ".pdf":
                orig_imgs, redact_imgs, redacted_fields = handle_pdf(uploaded_file.read())
                for idx, (orig, redacted) in enumerate(zip(orig_imgs, redact_imgs)):
                    tab_orig, tab_red = st.tabs([f"Page {idx+1} Original", f"Page {idx+1} Redacted"])
                    with tab_orig:
                        st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), use_column_width=True)
                    with tab_red:
                        st.image(cv2.cvtColor(redacted, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.download_button(
                    "Download all redacted pages as ZIP",
                    data=create_zip_from_images(redact_imgs),
                    file_name="redacted_pages.zip",
                    mime="application/zip"
                )
                redacted_fields_total.extend(redacted_fields)

            # ---------------- JSON ----------------
            elif ext == ".json":
                jdata = json.load(uploaded_file)
                redacted_json, redacted_fields = redact_json_with_faker_config(jdata, faker_config)
                st.subheader("Original JSON")
                st.json(jdata, expanded=True)
                st.subheader("Redacted JSON")
                st.json(redacted_json, expanded=True)
                st.download_button(
                    "Download redacted JSON",
                    data=json.dumps(redacted_json, indent=2),
                    file_name="redacted.json",
                    mime="application/json"
                )
                redacted_fields_total.extend(redacted_fields)

        # ---------------- API JSON ----------------
        if api_url:
            try:
                resp = requests.get(api_url)
                if resp.status_code == 200:
                    data = resp.json()
                    redacted_api_json, redacted_fields = redact_json_with_faker_config(data, faker_config)
                    st.subheader("API JSON (Original)")
                    st.json(data, expanded=True)
                    st.subheader("API JSON (Redacted)")
                    st.json(redacted_api_json, expanded=True)
                    st.download_button(
                        "Download API redacted JSON",
                        data=json.dumps(redacted_api_json, indent=2),
                        file_name="api_redacted.json",
                        mime="application/json"
                    )
                    redacted_fields_total.extend(redacted_fields)
                else:
                    st.error(f"API returned status code {resp.status_code}")
            except Exception as e:
                st.error(f"Failed to fetch API JSON: {e}")

        # ---------------- Redacted Fields ----------------
        if redacted_fields_total:
            section_block("Redacted Fields")
            st.write(redacted_fields_total)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9em;'>"
    "PII Shield &copy; 2025. Open Source. <br>Secure, offline, no data stored."
    "</div>", unsafe_allow_html=True
)
