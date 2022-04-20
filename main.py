
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


# cv2.waitKey(0)
# cv2.destroyAllWindows()
