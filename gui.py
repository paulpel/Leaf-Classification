import tkinter as tk
from tkinter import  E, W, Label, filedialog
from tkinter.filedialog import askopenfile
from unicodedata import name
from PIL import Image, ImageTk
from src.tools.load import LoadModel
import os

class GUI:

    def __init__(self) -> None:
        self.filename = None
        self.obj = LoadModel()
        self.obj.main()

        self.result = None
        
    
    def main(self):
        root = tk.Tk()
        self.text = tk.StringVar()
        self.text.set(f"Prediction: ")
        root.geometry("1300x460")  # Size of the window 
        root.title('LEAF CLASSIFICATION')

        label = Label(root, textvariable=self.text)
        label.grid(row=0, column=2, sticky= W, pady=5, padx=0)

        b1 = tk.Button(
            root, 
            text='Upload image for prediciton', 
            command =lambda:self.upload_file(),
            background='red')

        b2 = tk.Button(
            root, 
            text='Predict', 
            command=lambda:self.predict()
        )

        b1.grid(row=0, column=0, sticky= W, pady=5, padx=10)
        b2.grid(row=0, column=1, sticky= W, pady=5, padx=10)

        upload_im = Image.open('/Users/home/Programming/Git/Image Processing/upload.gif')
        upload_im = upload_im.resize((600, 400), Image.ANTIALIAS)
        img1 = ImageTk.PhotoImage(upload_im)
        photo1 = Label(image=img1)
        photo1.grid(row=1,column=0, sticky= W, pady=5, padx=10)

        leaves_im = Image.open('/Users/home/Programming/Git/Image Processing/leaves.jpeg')
        leaves_im = leaves_im.resize((600, 400), Image.ANTIALIAS)
        img2 = ImageTk.PhotoImage(leaves_im)
        photo2 = Label(image=img2)
        photo2.grid(row=1,column=1, sticky= W, pady=5, padx=10, columnspan=2)
        root.mainloop()  # Keep the window open

    def upload_file(self):
        global img
        
        f_types = [('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        self.result = self.obj.prepdata(filename)
        image = Image.open(filename)
        image = image.resize((600, 400), Image.ANTIALIAS) 
        img = ImageTk.PhotoImage(image)
        photo1 = Label(image=img)
        photo1.grid(row=1,column=0)

    def predict(self):
        global img3
        if self.result != None:
            path = os.path.join(
                '/Users/home/Programming/Git/Image Processing/src/data/full_data',
                self.result
            )
            full_p = os.path.join(path, os.listdir(path)[0])
            self.text.set(f'Prediction: {self.result}')
            im3 = Image.open(full_p)
            im3 = im3.resize((600, 400), Image.ANTIALIAS)
            img3 = ImageTk.PhotoImage(im3)
            photo3 = Label(image=img3)
            photo3.grid(row=1,column=1, sticky= W, pady=5, padx=10, columnspan=2)


if __name__ == '__main__':
    obj = GUI()
    obj.main()
    