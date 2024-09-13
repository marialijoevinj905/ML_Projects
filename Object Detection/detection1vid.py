import tkinter as tk
from tkinter import filedialog, Label
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('yolov10n.pt')

# Function to perform object detection on images
def detect_objects_in_image(file_path, label):
    image = cv2.imread(file_path)
    results = model(file_path)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_index = int(box.cls[0].item())
            label_text = f"{model.names[class_index]}: {box.conf[0]:.2f}"
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((label.winfo_width(), label.winfo_height()))
    img_tk = ImageTk.PhotoImage(image=img_pil)
    label.config(image=img_tk)
    label.image = img_tk

# Function to perform object detection on video
def detect_objects_in_video(file_path, label):
    cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf > 0.60:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{model.names[int(box.cls[0].item())]}: {box.conf[0]:.2f}"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (label.winfo_width(), label.winfo_height()), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label.config(image=img_tk)
        label.image = img_tk
        label.update()
    cap.release()

# Function to open image file and start detection
def open_image(label):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        detect_objects_in_image(file_path, label)

# Function to open video file and start detection
def open_video(label):
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
    if file_path:
        detect_objects_in_video(file_path, label)

# Initialize the main window
root = tk.Tk()
root.title("Object Detection with YOLO")
root.geometry("800x600")

# Label for displaying image or video
display_label = Label(root)
display_label.pack(fill=tk.BOTH, expand=True)

# Buttons for selecting image or video
btn_img = tk.Button(root, text="Detect in Image", command=lambda: open_image(display_label), height=2, width=20)
btn_img.pack(pady=10)

btn_vid = tk.Button(root, text="Detect in Video", command=lambda: open_video(display_label), height=2, width=20)
btn_vid.pack(pady=10)

root.mainloop()
