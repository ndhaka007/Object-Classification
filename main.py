import cv2
#import dlib
import time
import numpy as np
from keras.models import load_model

# Path to the Keras Model
model = load_model('new_weights.h5')
# Label order
label = ["A", "B", "C", "V", "W"]

# Initalize the webcam 0 is the web cam ID
cap = cv2.VideoCapture(0)
# will be used to dislpay FPS
start_time = time.time()
print ("Hit q to stop the code")
while(True):
    # Get frame form webcam
    ret, frame = cap.read()
	# Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Resize image
    I_crop = cv2.resize(gray, (96,96))
	# Equalizing the histogram is a very important part to improve performance
    I_crop = cv2.equalizeHist(I_crop)
	# If you will not convert image to float then all the values on dividing will either be zero or one
    I_crop = I_crop.astype('float32')
    I_crop /= 255
	# This is the shape of the INPUT in keras Model
    I_crop = I_crop.reshape(1, 96, 96, 1)
	# Forward Pass
    output = model.predict(I_crop)
    loc = (400,20)
	# Put label on Image
    emotion = label[np.argmax(output)] + ' ' + str(int(100*np.max(output))) + '%'
    cv2.putText(frame, emotion, loc, cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,0), 1)
	
    # Display FPS
    FPS = 'FPS :' + str(1/(time.time() - start_time))
    start_time = time.time()
    cv2.putText(frame, FPS, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255), 2)

    # Display the resulting frame    
    cv2.imshow('frame',frame)    

    # Hit q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
