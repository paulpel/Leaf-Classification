from PIL import Image
import os, sys


class resizeImage:

    def  __init__(self) -> None:
        self.cwd = os.getcwd()
        self.exceptions = ['.DS_Store']

    def main(self):
        self.resize()
        
    def resize(self):
        path = os.path.join(self.cwd, 'src', 'data')
        dirs = os.listdir(path)
        
        for dir in dirs:
            path_dir = os.path.join(path, dir)
            try:
                for file in os.listdir(path_dir):
                    file_path = os.path.join(path_dir, file)
                    im = Image.open(file_path)
                    f, e = os.path.splitext(file_path)
                    imResize = im.resize((120,80), Image.ANTIALIAS)
                    imResize.save(f + e, 'JPEG', quality=90)
            except Exception as err:
                print(err)
