# This is a sample Python script.

import pandas as pd
import cv2


index = pd.read_csv('index.csv')
df = index.copy()
#image = cv2.imread('harry-potter/0001/001.jpg')
# print(df['path'])

for imgPath in df.itertuples():
    path = imgPath[1]
    # print(path)
    image = cv2.imread(path)
    image_norm = cv2.normalize(image, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
    # print('norm-'+path)
    # cv2.imshow('norm', image_norm)
    cv2.imwrite('norm-' + path, image_norm)
    # print('done')

# image2 = cv2.imread(df['path'])
# image_norm = cv2.normalize(image2, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
# cv2.imshow('og', image2)
# cv2.imshow('norm', image_norm)
# cv2.imwrite('harry-potter/norm1.jpg', image_norm)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(df.columns)

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
