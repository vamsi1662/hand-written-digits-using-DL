import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Load trained model (we will retrain quickly here)
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, verbose=0)

# GUI setup
width = 200
height = 200

window = tk.Tk()
window.title("Digit Recognizer")

canvas = tk.Canvas(window, width=width, height=height, bg='white')
canvas.pack()

image = Image.new("L", (width, height), 255)
draw = ImageDraw.Draw(image)

def draw_lines(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
    draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

canvas.bind("<B1-Motion>", draw_lines)

def predict():
    img = image.resize((28, 28))
    img = np.array(img)
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1, 28, 28)

    prediction = model.predict(img)
    result = np.argmax(prediction)

    label.config(text="Prediction: " + str(result))

def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, width, height], fill=255)
    label.config(text="Draw a digit")

btn_predict = tk.Button(window, text="Predict", command=predict)
btn_predict.pack()

btn_clear = tk.Button(window, text="Clear", command=clear)
btn_clear.pack()

label = tk.Label(window, text="Draw a digit", font=("Arial", 16))
label.pack()

window.mainloop()