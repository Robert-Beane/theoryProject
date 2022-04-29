import shutil
import os

# function based off: https://pynative.com/python-copy-files-and-directories/

src = '../theoryProject/other/0004/'

for file in os.listdir(src):
    source = src + file
    destination = src + "0" + file
    if os.path.isfile(source):
        shutil.copy(source, destination)
        print('copied', file)