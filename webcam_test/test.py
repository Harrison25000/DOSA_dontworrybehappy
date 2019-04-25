from keras.models import load_model
# import imutils
import cv2
import numpy as np
import sys

model_top = load_model("my_model.h5")
EMOTION_DICT = {1:"ANGRY", 2:"FEAR", 3:"HAPPY", 4:"NEUTRAL", 5:"SAD", 6:"SURPRISE"}

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
        emotion_label = top_pred[0].argmax() + 1
        return EMOTION_DICT[emotion_label]
    except:
        pass


def rerun(text, cap):
    while (True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Last Emotion was " + str(text), (95, 30), font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(img, "Press SPACE: FOR EMOTION", (5, 470), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(img, "Hold Q: To Quit", (460, 470), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        try:
            x, y, w, h = faces[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
        except:
            pass

        cv2.imshow("Image", img)

        if cv2.waitKey(1) == ord(' '):
            cv2.imwrite("test.jpg", img)
            text = return_prediction("test.jpg")
            first_run(text, cap)
            break

        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml.txt')

cap = cv2.VideoCapture(0)

def first_run(text, cap):
    while (True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Last Emotion was " + str(text), (95, 30), font, 1.0, (0,0,255), 2, cv2.LINE_AA)

        cv2.putText(img, "Press SPACE: FOR EMOTION", (5, 470), font, 0.7, (0,0,255), 2, cv2.LINE_AA)

        cv2.putText(img, "Hold Q: To Quit", (460, 470), font, 0.7, (0,0,255), 2, cv2.LINE_AA)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        try:
            x, y, w, h = faces[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
        except:
            pass

        cv2.imshow("Image", img)

        if cv2.waitKey(1) == ord(' '):
            cv2.imwrite("test.jpg", img)
            text = return_prediction("test.jpg")
            rerun(text, cap)
            break

        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

first_run("None", cap)
