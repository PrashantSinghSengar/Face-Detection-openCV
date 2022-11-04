#importing openCV
import cv2

#as cv2.imshow crashes on servers s/a google colab we use:
#from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt


#instead of cv2.imshow we are using matplotlib to print the image
def cv2_imshow(a, **kwargs):
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    return plt.imshow(a, **kwargs)
  
  #importing faceclassifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#importing Image
img = cv2.imread('image1.jpg')

#convering the image to Grayscale as the model only works on grayscale images
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #OpenCV works as BGR notation instead of RGB

#detecting the faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#drawing rectangels around the faces
for(x,y,w,h) in faces:
  cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

#printing the image
#in case of end device execution we can use this
#cv2.imshow('img', img)
#cv2.waitKey()
#in case of server executions need to use this:
cv2_imshow(img)
