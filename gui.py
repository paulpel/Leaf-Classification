import tkinter as tk
from tkinter import  E, W, Label, filedialog
from tkinter.filedialog import askopenfile
from unicodedata import name
from PIL import Image, ImageTk
from numpy import full
from src.tools.load import LoadModel
import os

class GUI:

    def __init__(self) -> None:
        self.filename = None
        self.obj = LoadModel()
        self.obj.main()

        self.result_g = None
        self.result_c = None
        self.current_m = 'greyscale'
        
    
    def main(self):
        self.root = tk.Tk()
        self.text = tk.StringVar()
        self.text.set(f"Result: ")
        self.root.geometry("1300x510")  # Size of the window 
        self.root.title('LEAF CLASSIFICATION')

        label = Label(self.root, textvariable=self.text)
        label.grid(row=0, column=3, pady=0, padx=10)

        b1 = tk.Button(
            self.root, 
            text='Upload image for prediciton', 
            command =lambda:self.upload_file(),
            background='red')

        b2 = tk.Button(
            self.root, 
            text='Greyscale CNN', 
            command=lambda:self.predict_g()
        )
        b3 = tk.Button(self.root,
            text='Close winodw',
            command=quit)
        
        b4 = tk.Button(
            self.root, 
            text='Color CNN', 
            command=lambda:self.predict_c()
        )

        b1.grid(row=0, column=0, pady=5, padx=10)
        b2.grid(row=0, column=1, pady=5, padx=10)
        b3.grid(row=2, column=0, columnspan=3)
        b4.grid(row=0, column=2, pady=5, padx=10)

        upload_im = Image.open('/Users/home/Programming/Git/Image Processing/upload.gif')
        upload_im = upload_im.resize((600, 400), Image.ANTIALIAS)
        img1 = ImageTk.PhotoImage(upload_im)
        photo1 = Label(image=img1)
        photo1.grid(row=1,column=0, sticky= W, pady=5, padx=10)

        leaves_im = Image.open('/Users/home/Programming/Git/Image Processing/leaves.jpeg')
        leaves_im = leaves_im.resize((600, 400), Image.ANTIALIAS)
        img2 = ImageTk.PhotoImage(leaves_im)
        photo2 = Label(image=img2)
        photo2.grid(row=1,column=1, sticky= W, pady=5, padx=10, columnspan=3)
        self.root.mainloop()  # Keep the window open

    def upload_file(self):
        global img
        
        f_types = [('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        self.result_g, self.result_c = self.obj.prepdata(filename)
        image = Image.open(filename)
        image = image.resize((600, 400), Image.ANTIALIAS) 
        img = ImageTk.PhotoImage(image)
        photo1 = Label(image=img)
        photo1.grid(row=1,column=0)

    def predict_g(self):
        global img3
        if self.result_g != None:
            path1 = os.path.join(
                '/Users/home/Programming/Git/Image Processing/src/data/full_data/',
                self.result_g
            )
            
            full_p1 = os.path.join(path1, os.listdir(path1)[0])

            im3 = Image.open(full_p1)
            im3 = im3.resize((600, 400), Image.ANTIALIAS)
            img3 = ImageTk.PhotoImage(im3)
            photo3 = Label(image=img3)
            photo3.grid(row=1,column=1, sticky= W, pady=5, padx=10, columnspan=3)

            self.text.set(
                f'\nResult(grey): {self.result_g}')

    def predict_c(self):
        global img4
        if self.result_c != None:
            path2 = os.path.join(
                    '/Users/home/Programming/Git/Image Processing/src/data/full_data/',
                    self.result_c
                )
            full_p2 = os.path.join(path2, os.listdir(path2)[0])
            im4 = Image.open(full_p2)
            im4 = im4.resize((600, 400), Image.ANTIALIAS)
            img4 = ImageTk.PhotoImage(im4)
            photo4 = Label(image=img4)
            photo4.grid(row=1,column=1, sticky= W, pady=5, padx=10, columnspan=3)

            self.text.set(
                f'\nResult(color): {self.result_g}')

if __name__ == '__main__':
    obj = GUI()
    obj.main()
    