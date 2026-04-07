import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from time import time

classID = 1     # 0: fake, 1: real
offsetPercentageW = 10
offsetPercentageH = 20
confidence = 0.8
camWidth, camHeight = 640, 480
floatingPoint = 6
save = True
blurThreshold = 35
outputFolderImagesPath = 'Datasets/images/val'
outputFolderLabelsPath = 'Datasets/labels/val'
debug = False


cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)
detector = FaceDetector()

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img, draw=False)
    imgOut = img.copy()
    listBlur = [] 
    listInfo = []

    if bboxs:
        for bbox in bboxs:
            x,y,h,w = bbox["bbox"]
            score = bbox["score"][0]
            # print(score)
            # print(f"x:{x} y:{y} height: {h} width: {w}")

            if score > confidence:

                offsetW = (offsetPercentageW / 100)*w
                x = int(x - offsetW)
                w = int(w + offsetW*2)

                offsetH = (offsetPercentageH / 100)*h
                y = int(y - offsetH*3)
                h = int(h + offsetH*3.5)

                if x < 0: x = 0
                if y < 0: y = 0
                if h < 0: h = 0
                if w < 0: w = 0

                imgFace = img[y:y+h,x:x+w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace,cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                ih, iw, _ = img.shape
                xc,yc = x+w/2, y+h/2 
                xcn, ycn = round(xc/iw, floatingPoint) , round(yc/ih, floatingPoint)
                wn,hn = round(w/iw, floatingPoint), round(h/ih, floatingPoint)

                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1 : hn = 1

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                cv2.rectangle(imgOut, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score*100)} Blur: {blurValue}', (x,y-20), scale=2, thickness=3)

                if debug:
                    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score*100)} Blur: {blurValue}', (x,y-20), scale=2, thickness=3)

        if save:
            if all(listBlur) and listBlur != []:
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0] + timeNow[1]
                cv2.imwrite(f"{outputFolderImagesPath}/{timeNow}.jpg", img)

                for info in listInfo:
                    f = open(f"{outputFolderLabelsPath}/{timeNow}.txt", "a")
                    f.write(info)
                    f.close()


    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)
