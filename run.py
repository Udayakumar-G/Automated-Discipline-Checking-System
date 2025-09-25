import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ===========================================
# Load models and classes
# ===========================================
def load_model_and_classes(model_path, classes_file):
    model = load_model(model_path)
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return model, classes

dress_model, dress_classes = load_model_and_classes("dress_model.h5", "dress_classes.txt")
grooming_model, grooming_classes = load_model_and_classes("grooming_model.h5", "grooming_classes.txt")
posture_model, posture_classes = load_model_and_classes("posture_model.h5", "posture_classes.txt")
id_model, id_classes = load_model_and_classes("id_model.h5", "id_classes.txt")

# ===========================================
# Prediction function
# ===========================================
def predict_all(frame):
    img = cv2.resize(frame, (128,128))
    img_array = img.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = {}
    preds["dresscode"] = dress_classes[np.argmax(dress_model.predict(img_array))]
    preds["bear"] = grooming_classes[np.argmax(grooming_model.predict(img_array))]
    preds["posture"] = posture_classes[np.argmax(posture_model.predict(img_array))]
    preds["card"] = id_classes[np.argmax(id_model.predict(img_array))]

    return preds

# ===========================================
# Real-time webcam detection
# ===========================================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    preds = predict_all(frame)

    # Display predictions on the frame
    y0, dy = 30, 30
    for i, (k,v) in enumerate(preds.items()):
        y = y0 + i*dy
        cv2.putText(frame, f"{k}: {v}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Real-Time Discipline Check", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
