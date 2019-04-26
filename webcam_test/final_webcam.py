from keras.models import load_model
# import imutils
import cv2
import numpy as np
import sys

model_top = load_model("SHOW_ME_YOUR_EMOTIONS.h5")
EMOTION_DICT = {0:"ANGRY", 1:"HAPPY", 2:"SAD", 3:"SURPRISE"}
COLOUR_DICT = {"ANGRY":(0, 0, 255), "HAPPY":(153, 255, 51), "SAD":(255, 153, 153), "SURPRISE":(0,128,255)}

def return_prediction(path):
    # converting image to gray scale and save it
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path, gray)


    # detect face in image, crop it then resize it then save it
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml.txt')
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try:
        (x, y, w, h) = faces[0]
        face_clip = img[y:y + h, x:x + w]
        cv2.imwrite(path, cv2.resize(face_clip, (48, 48)))
    except:
        pass

    # read the processed image then make prediction and display the result
    read_image = cv2.imread(path, 0)
    try:
        read_image = np.asarray(read_image).reshape(1,48,48,1)
        read_image_final = read_image / 255.0
        # VGG_Pred = model_VGG.predict(read_image_final)
        # VGG_Pred = VGG_Pred.reshape(1, VGG_Pred.shape[1] * VGG_Pred.shape[2] * VGG_Pred.shape[3])
        top_pred = model_top.predict(read_image_final)
        print(top_pred)
        emotion_label = top_pred.argmax()
        return EMOTION_DICT[emotion_label]
    except:
        pass

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml.txt')

cap = cv2.VideoCapture(0)

def first_run(text, cap):
    while (True):

        cap = cv2.VideoCapture(0)
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        try:
            emotion = cv2.putText(img, "Last Emotion was " + str(text), (95, 30), font, 1.0, COLOUR_DICT[text], 2, cv2.LINE_AA)
        except:
            pass 

        try:
            x, y, w, h = faces[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), COLOUR_DICT[text], 2)
        except:
            pass


        cv2.putText(img, "Press Q: To Quit", (100, 200), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        cv2.imshow("Image", img)
        cv2.imwrite("test.jpg", img)
        try:
            text = return_prediction("test.jpg")
        except:
            pass

        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

first_run("None", cap)
