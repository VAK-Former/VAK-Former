import cv2

img1 = cv2.imread("E:/USV/mmsegmentation/vis_img/6.2.png")   # base image
img2 = cv2.imread("E:/USV/mmsegmentation/vis_img/6.1.png")   # overlay image

alpha = 0.5  # transparency
overlay = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)

cv2.imwrite("overlay.png", overlay)
