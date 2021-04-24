import NeuralNetwork as nn
import numpy as np
import random
from PIL import Image
from PIL import ImageOps
import cv2
from resizeimage import resizeimage
import speech_recognition as sr
import psutil
epochs = 20
lr = 0.1
import threading
import math
import time
import winsound
import pyttsx3
from datetime import datetime


open_menu = False
log = False
square_root = False
black_and_white = False
sumCommand = False
primeCommand = False
sineCommand = False
cosCommand = False
factorialCommand = False

maxNumber = 0

maxImages = 10

def Prime_Method(number):

    n = number
    count = 0
    for i in range(2, int(n - 1)):
        if int(n) % i == 0:
            count += 1

    print(count)
    if count == 0:
        return True
    else:
        return False

def SpeechRecognition():
    global open_menu, log, square_root, black_and_white, sumCommand, primeCommand, sineCommand, cosCommand, factorialCommand


    engine = pyttsx3.init()
    engine.say('Hello Mike, I am Bruce. How can I help you?')
    engine.runAndWait()



    while True:
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                audio = r.listen(source, phrase_time_limit=3)

            word = r.recognize_google(audio)
            print(word)
            if word == 'open the menu' or word == 'open the menu please' or word == 'please open the menu'\
                    or word == 'open the menu again' or word == 'open the menu again please' or word == 'please open the menu again' or word == 'on the menu' or word == 'the menu':

                engine.say('Opening the menu')
                engine.runAndWait()
                open_menu = True

            elif word == 'close the menu' or word == 'close it' or word == 'closet':

                engine.say('Closing the menu')
                engine.runAndWait()
                open_menu = False

            elif word == 'log' or word == 'lock':

                engine.say('Logarithm mode is on')
                engine.runAndWait()

                log = True

            elif word == 'normal' or word == 'back' or word == 'default' or word == 'revolt' or word == 'before':

                engine.say('Default mode is on')
                engine.runAndWait()

                black_and_white = False
                log = False
                square_root = False
                sumCommand = False

            elif word == 'root' or word == 'rude' or word == 'route':

                engine.say('Square root mode is on')
                engine.runAndWait()

                square_root = True

            elif word == 'thanks' or word == 'thank you':

                rnd = random.randint(0, 3)

                text = ''

                if rnd == 0:
                    text = 'Your welcome Mike'
                elif rnd == 1:
                    text = 'Your welcome'
                elif rnd == 2:
                    text = 'It is my honor to help you'
                elif rnd == 3:
                    text = 'My pleasure'

                engine.say(text)
                engine.runAndWait()

            elif word == 'time' or word == 'what\'s the time':
                now = datetime.now()
                current_time = now.strftime("%H:%M")

                engine.say(current_time)
                engine.runAndWait()

            elif word == 'Bruce' or word == 'Ruth' or word == 'Roar' or word == 'bruise' or word == 'growth':
                engine.say('Yes?')
                engine.runAndWait()

            elif word == 'black and white':
                black_and_white = True
                engine.say('Convert to black and white')
                engine.runAndWait()

            elif word == 'sum' or word == 'sun' or word == 'from' or word == 'Sam' or word == 'some' or word == 'thumb' or word == 'song':
                sumCommand = True
                engine.say('Calculating the sum from 1 to ' + str(maxNumber))
                engine.runAndWait()

            elif word == 'prime' or word == "rhyme" or word == "Prime":
                primeCommand = True

                if Prime_Method(maxNumber) == True:
                    engine.say(str(maxNumber) + ' is a prime number')
                else:
                    engine.say(str(maxNumber) + ' is not a prime number')
                engine.runAndWait()

            elif word == 'sine' or word == 'sin' or word == 'fine' or word == 'sign':
                sineCommand = True

                engine.say('Sine mode on')
                engine.runAndWait()

            elif word == 'cosine' or word == 'cos' or word == 'call sign':
                cosCommand = True
                engine.say('Cosine mode on')
                engine.runAndWait()

        except:
            pass




def MainProgram():
    global open_menu, log, square_root, black_and_white, maxNumber, sumCommand, primeCommand, sineCommand, cosCommand, factorialCommand
    class data_class:
        def __init__(self, input, target):
            self.input = input
            self.target = target



    def initialize_parameters_he(layers_dims):
        # np.random.seed(3)
        parameters = {}
        L = len(layers_dims) - 1
        for l in range(1, L + 1):
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
                2. / layers_dims[l - 1])
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        return parameters


    weights = initialize_parameters_he([784, 500, 11])

    weights['W2'] = np.loadtxt('W2.txt')
    weights['W1'] = np.loadtxt('W1.txt')

    weights['b2'] = np.loadtxt('b2.txt')
    weights['b1'] = np.loadtxt('b1.txt')

    weights['b2'] = weights['b2'].reshape(11, 1)
    weights['b1'] = weights['b1'].reshape(500, 1)

    threshVar = 100


    def maximum(array):
        nmax = -30000
        position = 0
        for i in range(0, len(array)):
            if array[i] > nmax:
                nmax = array[i]
                position = i
        return position


    def Main(im):
        def GetFromImage(im):
            data = im.getdata()

            new_data = []
            new_data += [(data[x] / 255) for x in range(0, 784)]

            return new_data

        INPUT = GetFromImage(im)
        inputImg = np.asarray(INPUT)
        inputImg = inputImg.reshape(784, 1)

        hidden1 = nn.create_hidden(weights['W1'], inputImg, weights['b1'], activation='sigmoid', dropout=False)
        output = nn.create_hidden(weights['W2'], hidden1, weights['b2'], activation='sigmoid', dropout=False)

        return output


    cap = cv2.VideoCapture(0)

    digit = 0
    accuracy = 0


    def PredictFrame(frame):
        global digit
        global accuracy
        img = Image.fromarray(frame)
        thresh = threshVar
        fn = lambda x: 255 if x > thresh else 0
        r = img.convert('L').point(fn, mode='1')
        inverted_image = ImageOps.invert(r.convert('RGB'))

        thumb = inverted_image.resize((22, 22), Image.ANTIALIAS)
        img2 = resizeimage.resize_contain(thumb, [22, 22], bg_color=(0, 0, 0))

        img_with_border = ImageOps.expand(img2, border=3, fill='black')

        # img_with_border.save('image2.png')

        im = img_with_border.convert('1')

        output = Main(im)
        digit = maximum(output)
        accuracy = np.max(output) * 100

        return digit, accuracy


    def SaveFrame(frame):
        global digit
        img = Image.fromarray(frame)
        thresh = threshVar
        fn = lambda x: 255 if x > thresh else 0
        r = img.convert('L').point(fn, mode='1')
        inverted_image = ImageOps.invert(r.convert('RGB'))

        thumb = inverted_image.resize((22, 22), Image.ANTIALIAS)
        img2 = resizeimage.resize_contain(thumb, [22, 22], bg_color=(0, 0, 0))

        img_with_border = ImageOps.expand(img2, border=3, fill='black')

        # img_with_border.save('image2.png')

        im = img_with_border.convert('1')

        im.save('img.png')


    BORDER = 40
    OFFSET = 2

    VALUES = []

    menu_color = (48, 48, 48)  # 70 179 70
    loop = 0

    battery_image = cv2.imread('batteryIcon2.png')
    rowsBattery, colsBattery, channelsBattery = battery_image.shape

    def sortCnts(cnts):

        value = []
        for index in range(0, len(cnts)):
            value.append(cnts[index][0])


        for index in range(0, len(cnts)):
            for j in range(index + 1, len(cnts)):
                if value[index] > value[j]:

                    temp2 = value[index]
                    value[index] = value[j]
                    value[j] = temp2

                    temp3 = cnts[index]
                    cnts[index] = cnts[j]
                    cnts[j] = temp3

        return cnts


    while True:
        try:
            maxNumber = _eval
        except:
            pass

        battery = psutil.sensors_battery()

        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_clean_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, clean_frame_bw = cv2.threshold(gray_clean_frame, threshVar, 225, cv2.THRESH_BINARY)


        (thresh, blackAndWhiteImage) = cv2.threshold(gray, threshVar, 255, cv2.THRESH_BINARY)

        cv2.threshold(gray, threshVar, 255, cv2.THRESH_BINARY, gray)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


        digitCnts = []

        coords1 = []
        coords2 = []


        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if (w >= 100 and w <= 290) and h >= 5:
                x1 = x + w
                y1 = y + h

                digitCnts.append([x, x1, y, y1])

                if len(digitCnts) <= 3:
                    if open_menu:
                        cv2.rectangle(frame, (x - BORDER, y - BORDER), (x1 + BORDER, y1 + BORDER), (255, 255, 255), 2)


        if len(digitCnts) <= 3:
            digitCnts = sortCnts(digitCnts)

            for i in range(0, len(digitCnts)):

                coords1.append((digitCnts[i][0] - BORDER + OFFSET, digitCnts[i][2] - BORDER + OFFSET))
                coords2.append((digitCnts[i][1] + BORDER - OFFSET, digitCnts[i][3] + BORDER - OFFSET))



            font = cv2.FONT_HERSHEY_SIMPLEX

            accurayList = []
            aaiwitpaiatltsi = []


            try:
                key = cv2.waitKey(1)
                crop_img = []

                i = 0

                while i < len(digitCnts):
                    crop_img.append(frame[coords1[i][1]:coords1[i][1] + (int)(coords2[i][1] - coords1[i][1]),
                           coords1[i][0]:coords1[i][0] + (int)(coords2[i][0] - coords1[i][0])])


                    i+=1

                i = 0
                _eval = 0
                while i < len(digitCnts):
                    digit, accuracy = PredictFrame(crop_img[i])
                    if digit == 10:
                        digit = '+'
                    else:
                        digit = str(digit)

                    aaiwitpaiatltsi.append(digit)
                    accurayList.append(accuracy)
                    i += 1


                if len(digitCnts) == 3:
                    if aaiwitpaiatltsi[2] == '+':
                        temp = aaiwitpaiatltsi[2]
                        aaiwitpaiatltsi[2] = aaiwitpaiatltsi[1]
                        aaiwitpaiatltsi[1] = temp

                    elif aaiwitpaiatltsi[0] == '+':
                        temp = aaiwitpaiatltsi[0]
                        aaiwitpaiatltsi[0] = aaiwitpaiatltsi[1]
                        aaiwitpaiatltsi[1] = temp


                exp = ''
                for s in aaiwitpaiatltsi:
                    exp += s

                _eval = eval(exp)

                if sumCommand:
                    maxNumber = _eval
                    a1 = 1
                    an = maxNumber
                    s = ((a1 + an) * an) / 2
                    _eval = s

                if log:
                    _eval = math.log(_eval)
                elif square_root:
                    _eval = math.sqrt(_eval)

                if sineCommand:

                    rad = (_eval * math.pi) / 180

                    _eval = math.sin(rad)
                    maxNumber = rad

                if cosCommand:
                    rad = (_eval * math.pi) / 180

                    _eval = math.cos(rad)
                    maxNumber = rad

                if factorialCommand:
                    fact = math.factorial(_eval)

                    _eval = fact
                    maxNumber = fact


            except:
                pass


            blk = np.zeros(frame.shape, np.uint8)

            try:
                if open_menu:
                    cv2.rectangle(blk, (640 - loop, 0), (640, 500), menu_color, cv2.FILLED)
                    cv2.putText(frame,
                                str(round(_eval, 2)),
                                (685 - loop, 50),
                                font, 0.7,
                                (45, 45, 45),
                                2,
                                cv2.LINE_4)

                    i = 0
                    offset = 50
                    while i < len(digitCnts):
                        cv2.putText(frame,
                                    str(round(accurayList[i], 2)) + ' - ' + str(aaiwitpaiatltsi[i]),
                                    (650 - loop, 30 + offset),
                                    font, 0.4,
                                    (45, 45, 45),
                                    2,
                                    cv2.LINE_4)

                        offset += 20
                        i += 1


                    cv2.putText(frame,
                                str(battery.percent) + '%',
                                (665 - loop, 335),
                                font, 0.7,
                                (45, 45, 45),
                                2,
                                cv2.LINE_4)
            except:
                pass


        if open_menu == True:
            if loop < 100:
                res = cv2.addWeighted(frame, 1, blk, 1, .5)

                loop += 5
            else:
                res = cv2.addWeighted(frame, 1, blk, 1, .5)


            try:
                res[350:350 + rowsBattery, 670 - loop:670 + colsBattery - loop] = battery_image
            except:
                pass

            if key == ord('s'):
                SaveFrame(crop_img[0])


        if open_menu == True:
            if not black_and_white:
                cv2.imshow("Frame", res)
            else:
                cv2.imshow("Frame", blackAndWhiteImage)
        else:
            if not black_and_white:
                cv2.imshow("Frame", frame)
            else:
                cv2.imshow("Frame", blackAndWhiteImage)

        digitCnts.clear()
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

p1 = threading.Thread(target=MainProgram)
p2 = threading.Thread(target=SpeechRecognition)

p1.start()
p2.start()


