import shutil
import os

# function based off: https://pynative.com/python-copy-files-and-directories/

src = '../theoryProject/marvel/0001/'
des = '../theoryProject/copy/'

for file in os.listdir(src):
    source = src + file
    destination = des + "0" + file
    if os.path.isfile(source):
        shutil.copy(source, destination)
        print('copied', file)