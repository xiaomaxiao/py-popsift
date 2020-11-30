
from ZHSift import ZHSift
import cv2

img = cv2.imread('../correct_img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift  = cv2.SIFT_create()
kp1 ,des1 = sift.detectAndCompute(img,None)


print(des1[0])

zhsift = ZHSift(0,True)

kp1,des2 = zhsift.detectAndCompute(img)

print(des2[-1])
