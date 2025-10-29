import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("../models/mobilenetv2_base.h5")

labels = ["closed_eyes", "no_yawn", "open_eyes", "yawn"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare frame
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    preds = model.predict(img)
    label = labels[np.argmax(preds)]

    # Show prediction on screen
    cv2.putText(frame, f"State: {label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
