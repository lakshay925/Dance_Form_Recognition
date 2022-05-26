import tkinter as tk
from tkinter import filedialog
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from keras.models import load_model
from pyasn1.compat.octets import null

image_size = 500

my_w = tk.Tk()
my_w.geometry("700x700")  # Size of the window
my_w.title('DANCE FORM RECOGNITION')
my_font1 = ('times', 18, 'bold')
my_font2 = ('times', 12, 'bold')

l1 = tk.Label(my_w, text='Upload and Test Photo', width=30, font=my_font1)
l1.grid(row=1, column=2)
b1 = tk.Button(my_w, text='Upload File',
   width=10, command=lambda: upload_file())
b1.grid(row=2, column=2)
b2 = tk.Button(my_w, text='Test File',
   width=10, command=lambda: test_file())
b2.grid(row=3, column=2)

model = load_model("model.h5") #model_loaded_from_file

def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(image_size, image_size))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor

def upload_file():
    global img
    global test_data
    f_types = [('Jpg Files', '*.jpg')]
    filepath = filedialog.askopenfilename(filetypes=f_types)
    test_data = load_image(filepath)
    img = Image.open(filepath)
    img_resized = img.resize((400, 400))  # new width & height
    img = ImageTk.PhotoImage(img_resized)
    b3 = tk.Button(my_w, image=img)  # using Button
    b3.grid(row=4, column=2)


def test_file():
    if(test_data == null):
        b4 = tk.Button(my_w, text="SELECT IMAGE", font=my_font2, width=40)  # using Button
        b4.grid(row=5, column=2)
        return

    pred = model.predict(test_data)
    print("predictions-")
    print(pred)

    list = {0: 'Baharatnatyam', 1: 'Kathak', 2: 'Kathakali', 3: 'Kuchipudi', 4: 'Manipuri', 5: 'Mohiniyattam',
            6: 'Odissi', 7: 'Sattriya'}

    prediction = ""
    max_val = 0.6
    index = 0

    for i in pred[0]:
        if (i > max_val):
            max_val = i
            prediction = list[index]
        index = index + 1

    global final_text
    if(prediction == ""):
        final_text = "NO_MATCH"
    else:
        final_text = prediction + ' - ' + (str((max_val * 100))[0:5]) + "%"


    b4 = tk.Button(my_w, text=final_text, font=my_font2, width=40)  # using Button
    b4.grid(row=5, column=2)

my_w.mainloop()  # Keep the window open