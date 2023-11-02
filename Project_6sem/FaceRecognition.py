import cv2
import numpy as np
import os
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras

# import Face_Recognition

# sys.path.insert(0, "D:\Machine Learning (ML)\Facial Recognission\Face_Recognition")
# sys.path.append('/Facial Recognission/Face_Recognition')
from CollectFaceData import capture_face
from cnn import CNN


def predict_faces(prediction_model, name):
    cam = cv2.VideoCapture(0)

    model = cv2.CascadeClassifier(
        "D:\Machine Learning (ML)\Project_6sem\haarcascade_frontalface_alt.xml"
    )
    offset = 20
    # Read image from camera

    while True:
        success, img = cam.read()
        if not success:
            print("Cannot Read From Camera")
            return

        faces = model.detectMultiScale(img, 1.3, 5)
        i = 0
        for f in faces:
            i += 1
            x, y, w, h = f

            cropped_face = img[y - offset : y + h + offset, x - offset : x + w + offset]
            cropped_shape = cropped_face.shape
            # print(cropped_shape)
            if cropped_shape[0] > 100 and cropped_shape[1] > 100:
                cropped_face = cv2.resize(cropped_face, (100, 100))
                IMG_SIZE = 100

                # cropped_face = cropped_face.flatten().reshape(1, -1)
                print(cropped_face.shape)
                # Predict class
                gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                face = np.array(gray).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
                print(face.shape)
                prediction = prediction_model.predict(face)
                # y_predict = model.predict(X_test)
                output = prediction.argmax(axis=1)
                print(prediction)
                print(output)
                output = int(output)
                print(output)
                print(name)
                namePredicted = name[output]

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    namePredicted,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            # cv2.imshow("Cropped" + str(i), cropped_face)

        cv2.imshow("Image Window", img)

        key = cv2.waitKey(10)
        if key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


choice = input("Do you want to add a new face (y/n) : ")
if choice.capitalize() == "Y":
    print(choice)
    capture_face()


choice = input("Do you want to Recognize face (y/n) : ")
if choice.capitalize() == "Y":
    # data Prepration
    dataset_path = "D:\Machine Learning (ML)\Project_6sem\Image_data/"  # "D:\Machine Learning (ML)\Facial Recognission\Face_Recognition\data/"
    facedata = []
    lables = []
    name = {}
    i = 0

    for f in os.listdir(dataset_path):
        if f.endswith(".npy"):
            dataItem = np.load(dataset_path + f)
            # plt.imshow(dataItem[0])
            print(dataItem.shape)
            m = dataItem.shape[0]
            target = i * np.ones((m,))
            name[i] = f[:-4]
            facedata.append(dataItem)
            lables.append(target)
            i += 1

    # print(facedata.shape)
    # print(lables, lables.shape)
    print("lables", lables)
    output_range = len(name)
    print(output_range)
    xt = np.concatenate(facedata, axis=0)
    yt = np.concatenate(lables, axis=0)
    # for i in faced

    print(xt.shape)
    print(yt.shape)
    # print(name)

    X_train, X_test, y_train, y_test = train_test_split(
        xt, yt, test_size=0.2, random_state=42
    )
    IMG_SIZE = 100

    X_trainr = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_testr = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    print(X_trainr.shape[1:])

    choice = input("Do you want to Train Model (y/n) : ")
    if choice.capitalize() == "Y":
        model = CNN(X_trainr, y_train, X_testr, y_test, 30, output_range)
        model.save("facial_recognition_cnn_model.h5")

    # model = KNeighborsClassifier(n_neighbors=3)
    # print(model.fit(xt, yt))
    model = keras.models.load_model("facial_recognition_cnn_model.h5")
    print(choice)
    predict_faces(model, name)
