import tkinter as tk
from tkinter import ttk, filedialog, PhotoImage
from tkinter.font import Font
from PIL import Image, ImageTk, ImageSequence
import itertools
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import os
import threading
from keras.models import load_model

# Load your custom age recognition model
age_model = load_model('age_recognition_2.h5')

age_range_mapping = {
    0: "1-4 years",
    1: "5-10 years",
    2: "11-20 years",
    3: "21-30 years",
    4: "31-40 years",
    5: "41-50 years",
    6: "51-60 years",
    7: "61-70 years",
    8: "71+ years"
}

# Load the face detection model
modelFile = "res10_300x300_ssd_iter_140000 (1).caffemodel"
configFile = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

screenshot_counter = 0

# Retrieve image file paths from a specified folder
def get_image_paths(folder_path):
    return glob.glob(f'{folder_path}/**/*.jpg', recursive=True)

def preprocess_image(img):
    # Dynamically adjusts image size to maintain aspect ratio
    h, w = img.shape[:2]
    scale = 224 / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# Function for face detection using CNN (SSD)
def face_detection_dnn(img):
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))
    return faces

# Function to display bounding boxes on detected faces
def box_disp(pic, facesdet, r1, r2, age_predictions):
    for ((x, y, w, h), age_pred) in zip(facesdet, age_predictions):
        # Adjust coordinates
        x1, y1 = int(x * r2), int(y * r1)
        x2, y2 = int(x1 + w * r2), int(y1 + h * r1)

        # Draw rectangle and text
        cv2.rectangle(pic, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(pic, age_pred, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

def predict_age(cropped_faces):
    age_predictions = []
    for face in cropped_faces:
        if face is not None and face.size != 0:
            resized_face = cv2.resize(face, (60, 60))
            # Check if the image is not already grayscale
            if len(resized_face.shape) == 3 and resized_face.shape[2] == 3:
                gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = resized_face

            face_array = gray_face.astype('float32') / 255.0
            face_array = face_array.reshape(1, 60, 60, 1)  # Ensure the correct shape
            predicted_age = age_model.predict(face_array)

            # Get the index with the highest probability
            predicted_index = np.argmax(predicted_age)

            # Map the predicted index to the corresponding age range
            age_range = age_range_mapping.get(predicted_index, "Unknown")
            age_predictions.append(age_range)
        else:
            age_predictions.append("No face detected")
    return age_predictions

def picCheck(image_path):
    global screenshot_counter

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to read the image.")
    processed_image = preprocess_image(image)
    faces = face_detection_dnn(processed_image)

    scaling_factor_y = image.shape[0] / processed_image.shape[0]
    scaling_factor_x = image.shape[1] / processed_image.shape[1]

    cropped_faces = []
    for (x, y, w, h) in faces:
        x1, y1 = int(x * scaling_factor_x), int(y * scaling_factor_y)
        x2, y2 = int(x1 + w * scaling_factor_x), int(y1 + h * scaling_factor_y)
        face_image = image[y1:y2, x1:x2]
        cropped_faces.append(face_image)

    age_predictions = predict_age(cropped_faces)
    box_disp(image, faces, scaling_factor_x, scaling_factor_y, age_predictions)

    cv2.namedWindow('Facial Recognition', cv2.WINDOW_NORMAL)
    cv2.imshow("Facial Recognition", image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            screenshot_path = os.path.join("screenshots", f'screenshot_{screenshot_counter}.jpg')
            cv2.imwrite(screenshot_path, image)
            print("Screenshot saved to", screenshot_path)
            screenshot_counter = (screenshot_counter + 1) % 5
        elif key == ord('q') or key == 27:  # 'q' or 'ESC' key to quit
            break

    cv2.destroyAllWindows()

# Modified process_video function with age recognition
def process_video(video_path=None):
    global screenshot_counter
    cap = cv2.VideoCapture(0 if video_path is None else video_path)

    if not cap.isOpened():
        print("Error: Couldn't open the video stream or file.")
        return

    window_name = 'Age Recognition in Video'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while cap.isOpened():
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read a frame from the video.")
                break

            # Flip the frame horizontally to mirror it
            frame = cv2.flip(frame, 1)

            original_height, original_width = frame.shape[:2]
            processed_frame = preprocess_image(frame)
            scaling_factor_y = original_height / processed_frame.shape[0]
            scaling_factor_x = original_width / processed_frame.shape[1]

            # Detect faces and predict age
            faces = face_detection_dnn(processed_frame)
            cropped_faces = [frame[y:y+h, x:x+w] for x, y, w, h in faces]

            # Predict ages
            age_predictions = predict_age(cropped_faces)

            # Adjust face coordinates based on the original frame size and display age
            adjusted_faces = [(int(x * scaling_factor_x), int(y * scaling_factor_y), int(w * scaling_factor_x), int(h * scaling_factor_y)) for x, y, w, h in faces]
            box_disp(frame, adjusted_faces, 1, 1, age_predictions)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                screenshot_path = os.path.join("screenshots", f'screenshot_{screenshot_counter}.jpg')
                cv2.imwrite(screenshot_path, frame)
                print("Screenshot saved to", screenshot_path)
                screenshot_counter = (screenshot_counter + 1) % 5

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def threaded_pic_check(image_path):
    thread = threading.Thread(target=picCheck, args=(image_path,))
    thread.start()

def threaded_process_video(video_path=None):
    thread = threading.Thread(target=process_video, args=(video_path,))
    thread.start()

# Functions to open a file dialog for selecting an image/video file
def select_picture():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        threaded_pic_check(file_path)

def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.mkv")])
    if file_path:
        threaded_process_video(file_path)

def load_image(path):
    img = Image.open(path)
    img = img.resize((100, 100), Image.ANTIALIAS)
    return ImageTk.PhotoImage(img)

def create_ui():
    window = tk.Tk()
    window.title("Age Recognition Application")
    window.geometry("1000x700")  # Adjust size as per your requirement
    window.configure(bg='black')

    # Heading
    heading_label = tk.Label(window, text="Age Recognition Application", bg='black', fg='white', font=("Arial", 24))
    heading_label.pack(pady=20)  # Add some padding for the heading

    # Frames for each image and button
    frame1 = tk.Frame(window, bg='black')
    frame1.pack(side=tk.LEFT, padx=10, pady=10, fill='both', expand=True)

    frame2 = tk.Frame(window, bg='black')
    frame2.pack(side=tk.LEFT, padx=10, pady=10, fill='both', expand=True)

    frame3 = tk.Frame(window, bg='black')
    frame3.pack(side=tk.LEFT, padx=10, pady=10, fill='both', expand=True)

    # Function to update the GIF frame
    def update_gif(ind):
        frame = gif_frames[ind]
        ind += 1
        if ind == len(gif_frames):
            ind = 0
        label2.configure(image=frame)
        window.after(100, update_gif, ind)  # Adjust delay as needed

    # Load static images for the first and third frames
    image1 = Image.open('pics\\Mino-Tamby-Profile-Picture.jpg').resize((250, 250))
    photo_image1 = ImageTk.PhotoImage(image1)
    label1 = tk.Label(frame1, image=photo_image1)
    label1.image = photo_image1  # Keep a reference!
    label1.pack(pady=10)

    image3 = Image.open('pics\\photo-camera-icons-photo-camera-icon-design-illustration-photo-camera-simple-sign-photo-camera-logo-vector.jpg').resize((250, 250))
    photo_image3 = ImageTk.PhotoImage(image3)
    label3 = tk.Label(frame3, image=photo_image3)
    label3.image = photo_image3  # Keep a reference!
    label3.pack(pady=10)

    # Load the GIF for the second frame and create an iterator for its frames
    gif = Image.open('pics\\gify.gif')
    gif_frames = [ImageTk.PhotoImage(image.resize((250, 250))) for image in ImageSequence.Iterator(gif)]
    label2 = tk.Label(frame2, image=gif_frames[0])
    label2.image = gif_frames  # Keep a reference!
    label2.pack(pady=10)

    # Start the animation
    window.after(0, update_gif, 0)

    # Pack the labels centered in their frame
    label1.pack(side='top')  # Center the label within the top part of the frame
    label2.pack(side='top')
    label3.pack(side='top')

    # Buttons below images
    button_texts = ['Select Picture', 'Select Video', 'Real-time Check']
    commands = [select_picture, select_video, threaded_process_video]
    frames = [frame1, frame2, frame3]
    buttons = [tk.Button(frame, text=text, command=cmd, bg='#4caf50', fg='white') for frame, text, cmd in zip(frames, button_texts, commands)]
    for button in buttons:
        button.pack(fill='x', pady=5)

    window.mainloop()

create_ui()
