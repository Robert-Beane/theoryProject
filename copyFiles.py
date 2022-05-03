import shutil
import os

# function based off: https://pynative.com/python-copy-files-and-directories/

src = '../theoryProject/star-wars/0016/'

for file in os.listdir(src):
    source = src + file
    destination = src + "37" + file
    if os.path.isfile(source):
        shutil.copy(source, destination)
        print('copied', file)