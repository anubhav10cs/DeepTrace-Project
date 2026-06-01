import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms

import torch.nn as nn

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "mnsitmod.pth")



class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier= nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,10)

        )
    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)

        return x
    

model= MNISTModel()

model.load_state_dict(torch.load(MODEL_PATH,map_location="cpu"))

model.eval()

root=tk.Tk()
root.title("Digit Recognizer APP")

canvas_size=280
canvas=tk.Canvas(
    root,
    width=canvas_size,
    height=canvas_size,
    bg="black"
)

canvas.pack(pady=10)

image=Image.new("L",(canvas_size,canvas_size),color=0)

draw=ImageDraw.Draw(image)


def paint(event):
    x = event.x
    y = event.y

    r=10

    canvas.create_oval(
        x-r,
        y-r,
        x+r,
        y+r,
        fill="white",
        outline="white"
    )


    draw.ellipse(
        (x-r,y-r,x+r,y+r),
        fill=255
    )

canvas.bind("<B1-Motion>",paint)

result=tk.Label(
    root,
    text="Draw a digit",
    font=("Arial",16)
)

result.pack()



def predict():
    img= image.copy()
    img=img.resize((28,28))

    transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307),(0.3081))
    ])

    x= transform(img)

    x= x.unsqueeze(0)

    with torch.no_grad():
        output=model(x)

        probs= torch.softmax(output,dim=1)

        pred=torch.argmax(probs,dim=1).item()

        confidence= probs[0][pred].item()

        result.config(text=f"Prediction : {pred} ({confidence*100}) %")


def clear():
    canvas.delete("all")

    global image
    global draw

    image=Image.new("L",(canvas_size,canvas_size),color=0)

    draw=ImageDraw.Draw(image)

    result.config(
        text="Draw a Digit"
    )



predict_button=tk.Button(
    root,text="Predict",
    command=predict
)

predict_button.pack(pady=5)

clear_button= tk.Button(
    root,
    text="Clear",
    command= clear
)

clear_button.pack(pady=5)

root.mainloop()
