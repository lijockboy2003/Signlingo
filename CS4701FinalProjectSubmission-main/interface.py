import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


# Define the CNN architecture
class Signlingo(nn.Module):
    def __init__(self, num_classes):
        super(Signlingo, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(24 * 24 * 64, 512)
        self.fc2 = nn.Linear(512, 29)

    def forward(self, x):
        x = self.MaxPool(torch.relu(self.conv1(x)))
        x = self.MaxPool(torch.relu(self.conv2(x)))
        x = self.MaxPool(torch.relu(self.conv3(x)))
        # x = self.MaxPool(torch.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


possible_words = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "X",
    "Y",
    "W",
    "Z",
]

camera_running = False
vid = None
camera_after_id = None
signed_word = ""
model = torch.load("model.pth")


def next_word():
    return np.random.choice(possible_words)


def next_action():
    current_word = next_word()
    page2word.configure(text="Make the gesture for: " + current_word)
    return current_word


def open_camera(current_word):
    global vid, camera_running, camera_after_id, signed_word, model

    signed_word = np.random.choice(possible_words)
    identified_letter.configure(text="You are signing: " + signed_word)
    if not camera_running:
        return

    if vid is None or not vid.isOpened():
        vid = cv2.VideoCapture(0)
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ret, frame = vid.read()
    if ret:
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)
        label_widget.photo_image = photo_image
        label_widget.configure(image=photo_image)

        transform = transforms.Compose(
            [
                transforms.Resize((192, 192)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        input_image = transform(captured_image).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            output = model(input_image)
        predicted_class = torch.argmax(output).item()
        class_to_letter = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "DEL",
            5: "E",
            6: "F",
            7: "G",
            8: "H",
            9: "I",
            10: "J",
            11: "K",
            12: "L",
            13: "M",
            14: "N",
            15: "Nothing",
            16: "O",
            17: "P",
            18: "K",
            19: "R",
            20: "S",
            21: "Space",
            22: "T",
            23: "U",
            24: "V",
            25: "W",
            26: "X",
            27: "Y",
            28: "Z",
        }
        predicted_letter = class_to_letter[predicted_class]
        identified_letter.configure(text="You are signing: " + predicted_letter)

    camera_after_id = label_widget.after(10, open_camera, current_word)


def reset():
    global vid, camera_running, camera_after_id

    if camera_after_id is not None:
        label_widget.after_cancel(camera_after_id)

    if vid is not None:
        if vid.isOpened():
            vid.release()
        vid = None

    camera_running = False

    label_widget.configure(image=None)

    page2word.configure(text="")
    identified_letter.configure(text="")

    page1_frame.pack(fill="both", expand=True)
    page2_frame.pack_forget()


def page1():
    reset()
    page1_title.pack()
    page1_description.pack()
    page1_button.pack()


def page2():
    global camera_running
    page1_frame.pack_forget()
    page2_frame.pack(fill="both", expand=True)
    camera_running = True
    if camera_running:
        open_camera(next_action())


window = tk.Tk()
window.title("Signlingo")
window.geometry("500x700")

page1_frame = tk.Frame(window)
page2_frame = tk.Frame(window)

page1_title = tk.Label(page1_frame, text="Welcome to Signlingo", font=("Arial", 20))
page1_title.pack(pady=20)

page1_description = tk.Label(
    page1_frame,
    text="Learn sign language interactively with real-time feedback.",
    font=("Arial", 14),
)
page1_description.pack(pady=10)

page1_button = tk.Button(
    page1_frame, text="Start Learning", command=page2, font=("Arial", 14), bg="green"
)
page1_button.pack(pady=20)

page1_frame.pack(fill="both", expand=True)

page2_title = tk.Label(page2_frame, text="Sign Language Practice", font=("Arial", 20))
page2_title.pack(pady=20)

label_widget = tk.Label(page2_frame, height=350, width=350)
label_widget.pack(pady=10)

page2word = tk.Label(page2_frame, text="", font=("Arial", 14))
page2word.pack(pady=10)

identified_letter = tk.Label(page2_frame, text="", font=("Arial", 14))
identified_letter.pack(pady=10)

page2_button_frame = tk.Frame(page2_frame)
page2_button_frame.pack(pady=20)

page2next = tk.Button(
    page2_button_frame,
    text="Next Word",
    command=next_action,
    font=("Arial", 12),
    bg="blue",
)
page2next.pack(side=tk.LEFT, padx=10)

page2leave = tk.Button(
    page2_button_frame, text="Back to Home", command=page1, font=("Arial", 12), bg="red"
)
page2leave.pack(side=tk.LEFT, padx=10)

width, height = 350, 350

window.mainloop()
