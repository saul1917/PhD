import cv2

img1 = cv2.imread('ART_00272_otsu.jpg')
img2 = cv2.imread('ART_00283_kittler.jpg')

print(img1.size)
print(img2.size)
dst = cv2.addWeighted(img1,0.7,img2,0.3,0)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
