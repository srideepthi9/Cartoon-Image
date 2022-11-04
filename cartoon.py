import easygui
import matplotlib.pyplot as plt
import cv2

imgPath=easygui.fileopenbox() #upload image from library
readImg = cv2.imread(imgPath) #reading image as numbers (it will generatenumpy array)
readImg = cv2.cvtColor(readImg, cv2.COLOR_BGR2RGB) #converting image from BGR color space to RGB

grayImg = cv2.cvtColor(readImg, cv2.COLOR_BGR2GRAY) #first we convert image to gray image to extract edges

smooth = cv2.medianBlur(grayImg, 5) #smoothen image so that edges can be extracted easily and correctly

getEdge = cv2.adaptiveThreshold(smooth, 255, 
  cv2.ADAPTIVE_THRESH_MEAN_C, 
  cv2.THRESH_BINARY, 21, 10)
#getting edges
plt.imshow(getEdge, cmap='gray')
plt.show()

colorImage = cv2.bilateralFilter(readImg, 21, 300, 300)
#smooth colored image
cartoon = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
#masking colored image with edges to give cartoon look

plt.imshow(cartoon, cmap='gray')
plt.show()
plt.savefig('saved_figure.png')
