import cv2
import pyttsx3
import threading
from ultralytics import YOLO
import time

# دالة النطق بخيط منفصل
def speak(text):
    def _speak():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak).start()

# تحميل الموديل (خليه نسخة أصغر لو بطيء: yolov8n.pt)
model = YOLO("yolov8n.pt")

# الأشياء الخطرة
danger_objects = ["chair", "bench", "stairs", "door", "bicycle", "sofa", "table"]

# الكاميرا (IP Webcam أو 0 للكاميرا العادية)
ip_cam_url = "http://192.168.43.50:8080/video"  # عدّلي حسب جهازك
cap = cv2.VideoCapture(ip_cam_url)

spoken_objects = {}
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (320, 240))

    if frame_count % 2 == 0:
        results = model(frame)
        current_time = time.time()

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                if label in danger_objects:
                    last_spoken = spoken_objects.get(label, 0)
                    if current_time - last_spoken > 3:
                        spoken_objects[label] = current_time
                        speak(f"Warning! {label} ahead")  # نطق بخيط منفصل

    frame_count += 1
    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()