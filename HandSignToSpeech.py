
import multiprocessing
import numpy as np
import cv2
import tensorflow
import tensorflow.keras as tf
import math
import os
from gtts import gTTS
import speech_recognition as sr
import time
import playsound
from pydub.playback import play
from pydub import AudioSegment
AudioSegment.converter = "avprobe"
from moviepy.editor import AudioFileClip
import pyttsx3



# etiketleri labels.txt dosyasından alma
labels_path = "labels.txt"
labelsfile = open(labels_path, "r")

# initialize classes and read in lines until there are no more
classes = []
line = labelsfile.readline()
while line:
    # retrieve just class name and append to classes
    classes.append(line.split(" ", 1)[1].rstrip())
    line = labelsfile.readline()
# close label file
labelsfile.close()
print(classes)

# load the teachable machine model
model_path = "Model/keras_model.h5"
model = tf.models.load_model(model_path, compile=False)

cap = cv2.VideoCapture(0)

# width & height of webcam video in pixels -> adjust to your size
# adjust values if you see black bars on the sides of capture window
frameWidth = 1280
frameHeight = 540

# set width and height in pixels
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
# enable auto gain
cap.set(cv2.CAP_PROP_GAIN, 0)
tahminler = []
i = 0

folder = "Image-Test"
input_shape = (224, 224)


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # Görüntüyü [0, 1] aralığına normalize etme
    image = np.expand_dims(
        image, axis=0
    )  # Boyutu (1, height, width, channels) şeklinde genişletme
    return image


# Görüntüyü işleme ve tahmin alma
def predict_image(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    return predicted_class_label, confidence


# def TurnToSpeech(word):
#     tts = gTTS(text=word, lang="tr")
#     speech = "Ses.mp3"
#     tts.save(speech)
#     ses_dosyasi=AudioSegment.from_file(speech,format="mp3")
#     play(ses_dosyasi)




while True:
    # disable scientific notation for claarity
    np.set_printoptions(suppress=True)

    # Create the array of the right shape to feed into the keras model.
    # We are inputting 1x 224x224 pixel RGB image.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # capture image
    check, frame = cap.read()
    frame = cv2.flip(frame, 1)  ### flip komutu ile sağ ve sol yöndeki tersliği düzelt

    # crop to square for use with TM model
    margin = int(((frameWidth - frameHeight) / 2))
    square_frame = frame[0:frameHeight, margin : margin + frameHeight]
    # resize to 224x224 for use with TM model
    resized_img = cv2.resize(square_frame, (224, 224))
    # convert image color to go to model
    model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # turn the image into a numpy array
    image_array = np.asarray(model_img)
    # normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # load the image into the array
    data[0] = normalized_image_array

    # run the predictions

    # 5 snde 1 ss al
    # aldığın ss'in predictionunu al
    # predictionunu aldıktan sonra class'a ekle
    # ses olarak okut

    predictions = model.predict(data)

    # confidence threshold is 90%.
    conf_threshold = 90
    confidence = []
    conf_label = ""
    threshold_class = ""
    # create blach border at bottom for labels
    per_line = 2  # number of classes per line of text
    bordered_frame = cv2.copyMakeBorder(
        square_frame,
        top=0,
        bottom=30 + 15 * math.ceil(len(classes) / per_line),
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    # for each one of the classes
    for i in range(0, len(classes)):
        # scale prediction confidence to % and apppend to 1-D list
        confidence.append(int(predictions[0][i] * 100))
        # put text per line based on number of classes per line
        if i != 0 and not i % per_line:
            cv2.putText(
                img=bordered_frame,
                text=conf_label,
                org=(int(0), int(frameHeight + 25 + 15 * math.ceil(i / per_line))),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
            )
            conf_label = ""
        # append classes and confidences to text for label
        conf_label += classes[i] + ": " + str(confidence[i]) + "%; "
        # prints last line
        if i == (len(classes) - 1):
            cv2.putText(
                img=bordered_frame,
                text=conf_label,
                org=(
                    int(0),
                    int(frameHeight + 25 + 15 * math.ceil((i + 1) / per_line)),
                ),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
            )
            conf_label = ""
        # if above confidence threshold, send to queue
        if confidence[i] > conf_threshold:
            # speakQ.put(classes[i])
            threshold_class = classes[i]
    # add label class above confidence threshold
    cv2.putText(
        img=bordered_frame,
        text=threshold_class,
        org=(int(0), int(frameHeight + 20)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.9,
        color=(255, 255, 255),
    )

    # display and wait 1ms
    cv2.imshow("webcam goruntusu", bordered_frame)

    key = cv2.waitKey(1)
    if (key) == ord("q"):
        break

    elif (key) == ord("s"):
        tahmin = threshold_class
        image_path = f"{folder}/Image_{time.time()}.jpg"
        cv2.imwrite(image_path, bordered_frame)
        predicted_label, confident = predict_image(image_path)
        print(f"Image: {image_path}")
        print("Predicted Label:", predicted_label)
        tahminler.append(predicted_label)
        print(tahminler)

        cv2.putText(
            img=bordered_frame,
            text=f"{predicted_label}",
            org=(int(140), int(frameHeight + 20)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(255, 255, 255),
        )

        cv2.putText(
            img=bordered_frame,
            text="1-D",
            org=(int(180), int(frameHeight + 40)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
        )

        cv2.putText(
            img=bordered_frame,
            text="2-Y",
            org=(int(240), int(frameHeight + 40)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
        )

        cv2.imshow("Foto", bordered_frame)

        print(tahminler)
        cv2.waitKey(0)  # Kullanıcı bir tuşa basana kadar beklet
        cv2.destroyAllWindows()  # Tüm pencereleri kapat

        Sentence = "".join(tahminler)
        Sentence = str(Sentence)
        print(Sentence)
        print("Is Word Okey ?")
        print("Press 1 if  its ture")
        print("Press anything if its false")
        x = input("Choose: ")

        if x == "1":
        #  tts = gTTS(text=Sentence, lang="tr")
        #  speech = "Ses.mp3"
        #  tts.save(speech)
        #  ses_dosyasi=AudioSegment.from_file(speech,format="mp3")
        #  play(ses_dosyasi)
         txt_speech=pyttsx3.init()
         txt_speech.say(Sentence)
         txt_speech.runAndWait()
         
         break
