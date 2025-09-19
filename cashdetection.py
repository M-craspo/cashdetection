import cv2
import os
import time
import logging
from datetime import datetime
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

# ==========================
# CONFIG
# ==========================
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")

CAMERAS = {
    "waterway": os.getenv("CAMERA_WATERWAY"),
    "mivida": os.getenv("CAMERA_MIVIDA")
}

EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")
EMAIL_PASS = os.getenv("EMAIL_PASS")
COOLDOWN = int(os.getenv("COOLDOWN", 300))
CONF_THRES = float(os.getenv("CONF_THRES", 0.25))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ==========================
# EMAIL FUNCTION
# ==========================
def send_email(subject, body, image_path=None):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO

        if image_path and os.path.exists(image_path):
            _, img_encoded = cv2.imencode(
                '.jpg', cv2.imread(image_path),
                [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )
            msg.add_attachment(img_encoded.tobytes(),
                               maintype='image',
                               subtype='jpeg',
                               filename=os.path.basename(image_path))

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.send_message(msg)
            logging.info("📧 Email sent successfully")
    except Exception as e:
        logging.error(f"Email error: {e}")

# ==========================
# YOLO DETECTION
# ==========================
def detect_cash(model, frame, conf_thres=0.25):
    results = model(frame, conf=conf_thres)
    cash_detected = False

    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])].lower()
            conf = float(box.conf[0])
            if cls_name == "cash" and conf >= conf_thres:
                cash_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"Cash {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                logging.info(f"💰 Cash detected (conf={conf:.2f})")
    return cash_detected, frame

# ==========================
# MAIN LOOP
# ==========================
def main():
    logging.info("🚀 Starting Multi-Camera Cash Detection")

    try:
        model = YOLO(MODEL_PATH)
        logging.info("✅ YOLO model loaded")
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        return

    last_detection_time = {cam: 0 for cam in CAMERAS}

    while True:
        for name, url in CAMERAS.items():
            cap = cv2.VideoCapture(url)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                logging.warning(f"❌ No frame from camera {name}")
                continue

            cash_detected, processed_frame = detect_cash(model, frame, CONF_THRES)
            out_file = f"detection_{name}.jpg"
            cv2.imwrite(out_file, processed_frame)

            if cash_detected and (time.time() - last_detection_time[name] > COOLDOWN):
                send_email(
                    f"💰 Cash Detected in {name}",
                    f"Cash detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    out_file
                )
                last_detection_time[name] = time.time()

        time.sleep(3)  

if __name__ == "__main__":
    main()
