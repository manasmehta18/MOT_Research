import cv2
  
# path
# path = 'C:/Users/manas/Desktop/Yeet/unet/y.jpg'
path = 'C:/Users/manas/Desktop/Yeet/unet/data/masks/000139.jpg'
  
# Using cv2.imread() method
img = cv2.imread(path)
  
# Displaying the image
img = 255.0*img

cv2.imshow('image', img)
print(img)
cv2.waitKey(0) & 0xff  