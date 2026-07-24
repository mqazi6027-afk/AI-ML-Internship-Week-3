from tensorflow.keras.models import load_model
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os


# ==========================================
# MAIN WINDOW
# ==========================================

app = ttk.Window(
    title="😀 Facial Emotion Detector AI",
    themename="darkly",
    size=(900,600)
)

app.resizable(False,False)



# ==========================================
# VARIABLES
# ==========================================

image_path = None
display_image = None
detected_face = None



# ==========================================
# LOAD MODEL
# ==========================================
model_path = os.path.join(
    os.path.dirname(__file__),
    "../models/emotion_model.h5"
)

print("Looking for model at:", model_path)

if os.path.exists(model_path):
    try:
        model = load_model(model_path)
        print(model.summary())
        model_status = "AI Model Loaded ✅"
        print("Model loaded successfully!")
    except Exception as e:
        model = None
        model_status = "Model Loading Failed ❌"
        print("MODEL ERROR:", e)
else:
    model = None
    model_status = "Model Missing ⚠️"
    print("Model file not found!")

# ==========================================
# EMOTION LABELS
# ==========================================

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# ==========================================
# HEADER
# ==========================================

header = ttk.Frame(app)

header.pack(
    fill=X,
    pady=5
)



title = ttk.Label(
    header,
    text="😀 Facial Emotion Detector",
    font=("Segoe UI",22,"bold"),
    bootstyle="info"
)

title.pack()



subtitle = ttk.Label(
    header,
    text="AI Facial Emotion Recognition System",
    font=("Segoe UI",11)
)

subtitle.pack()



# ==========================================
# MAIN AREA
# ==========================================

main_frame = ttk.Frame(app)

main_frame.pack(
    fill=BOTH,
    expand=True,
    padx=15,
    pady=5
)



# LEFT PANEL

left_frame = ttk.Labelframe(
    main_frame,
    text="Image Preview",
    padding=10
)

left_frame.pack(
    side=LEFT,
    fill=BOTH,
    expand=True,
    padx=5
)



image_label = ttk.Label(
    left_frame,
    text="No Image Selected",
    font=("Segoe UI",13)
)

image_label.pack(
    expand=True
)



# RIGHT PANEL

right_frame = ttk.Labelframe(
    main_frame,
    text="Prediction",
    padding=10
)

right_frame.pack(
    side=RIGHT,
    fill=BOTH,
    expand=True,
    padx=5
)



emotion_label = ttk.Label(
    right_frame,
    text="Emotion: ---",
    font=("Segoe UI",16,"bold")
)

emotion_label.pack(pady=15)



confidence_label = ttk.Label(
    right_frame,
    text="Confidence: ---",
    font=("Segoe UI",13)
)

confidence_label.pack(pady=5)



# HISTORY

history_title = ttk.Label(
    right_frame,
    text="History",
    font=("Segoe UI",12,"bold")
)

history_title.pack(pady=10)



history_box = ttk.Listbox(
    right_frame,
    height=6
)

history_box.pack()



# ==========================================
# PREDICT FUNCTION
# ==========================================
def predict_emotion():
    global detected_face

    print("1. Predict button clicked")

    if detected_face is None:
        print("2. No detected face")
        status.config(text="Please upload image first ❌")
        return

    print("3. Face found")

    if model is None:
        print("4. Model is None")
        status.config(text="Model not loaded ❌")
        return

    print("5. Model loaded")

    try:
        face = cv2.resize(detected_face, (48, 48))
        print("6. Face resized")

        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        print("7. Shape:", face.shape)

        prediction = model.predict(face, verbose=0)
        print(prediction)
        print(prediction.shape)

        print("8. Prediction:", prediction)

        index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        emotion = emotion_labels[index]

        emotion_label.config(text=f"Emotion: {emotion}")
        confidence_label.config(text=f"Confidence: {confidence:.2f}%")
        history_box.insert(END, f"{emotion} - {confidence:.2f}%")

        status.config(text="Prediction completed ✅")

        print("9. Done")

    except Exception as e:
        print("ERROR:", e)
        status.config(text=str(e))
    # ==========================================
# UPLOAD IMAGE FUNCTION
# ==========================================

def upload_image():

    global image_path, display_image, detected_face


    image_path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files","*.jpg *.jpeg *.png")
        ]
    )


    if image_path:


        img = cv2.imread(image_path)


        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )


        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )


        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )


        if len(faces) == 0:

            status.config(
                text="No face detected ❌"
            )

            return



        for (x,y,w,h) in faces:


            detected_face = gray[
                y:y+h,
                x:x+w
            ]


            cv2.rectangle(
                img,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                3
            )

            break



        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )


        img = Image.fromarray(img)


        img.thumbnail(
            (350,350)
        )


        display_image = ImageTk.PhotoImage(
            img
        )


        image_label.config(
            image=display_image,
            text=""
        )


        status.config(
            text="Face detected ✅ Click Predict"
        )



# ==========================================
# CLEAR FUNCTION
# ==========================================

def clear_all():

    global detected_face, display_image


    detected_face = None
    display_image = None


    image_label.config(
        image="",
        text="No Image Selected"
    )


    emotion_label.config(
        text="Emotion: ---"
    )


    confidence_label.config(
        text="Confidence: ---"
    )


    history_box.delete(
        0,
        END
    )


    status.config(
        text="Cleared"
    )



# ==========================================
# BUTTONS
# ==========================================

button_frame = ttk.Frame(app)

button_frame.pack(
    pady=10
)



upload_btn = ttk.Button(
    button_frame,
    text="📂 Upload Image",
    bootstyle="primary",
    width=18,
    command=upload_image
)

upload_btn.grid(
    row=0,
    column=0,
    padx=5
)



predict_btn = ttk.Button(
    button_frame,
    text="🧠 Predict Emotion",
    bootstyle="success",
    width=18,
    command=predict_emotion
)

predict_btn.grid(
    row=0,
    column=1,
    padx=5
)



clear_btn = ttk.Button(
    button_frame,
    text="🗑 Clear",
    bootstyle="warning",
    width=15,
    command=clear_all
)

clear_btn.grid(
    row=0,
    column=2,
    padx=5
)



exit_btn = ttk.Button(
    button_frame,
    text="❌ Exit",
    bootstyle="danger",
    width=15,
    command=app.destroy
)

exit_btn.grid(
    row=0,
    column=3,
    padx=5
)



# ==========================================
# STATUS BAR
# ==========================================

status = ttk.Label(
    app,
    text=model_status,
    font=("Segoe UI",12,"bold"),
    anchor=W,
    bootstyle="secondary"
)

status.pack(
    fill=X,
    side=BOTTOM
)



# ==========================================
# RUN APPLICATION
# ==========================================

app.mainloop()