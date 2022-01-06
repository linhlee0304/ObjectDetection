import cv2
import numpy as np
import os
import random
import array
# use webcam
# cap = cv2.VideoCapture(0)

#use img
path = os.path.join('images/img (45).jpg')
img = cv2.imread(path)

whT = 416

confThresold = 0.2 #Nếu xác suất của vật thể nhỏ hơn 0.5 thì model sẽ loại bỏ vật thể đó

nms_threshold = 0.3 #Nếu có nhiều box chồng lên nhau, và vượt quá giá trị 0.3 (tổng diện tích chồng nhau)
                    # thì  1 trong 2 box sẽ bị loại bỏ.


#Đọc file coco.names và gán các nhãn vào mảng classNames
classesFile = 'coco.names'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rsplit('\n')

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print(net)
def findOjects(output, img):
    hT, wT, cT = img.shape

    bbox = []
    classIds = []
    confs = []
    wr = 0
    ri = 0
    # count = 0
    for output in outputs:
        for det in output:
            scores = det[5:] #Lấy từ phần tử thứ 5(object 0) -> object 80
            # print(scores)
            classId = np.argmax(scores) #Lấy vị trí giá trị lớn nhất trong mảng
            # print(classId)
            confidence = scores[classId] #gán giá trị
            # print(confidence)
            if confidence >= confThresold: #so sánh confidence vs conf mong muốn
                w,h = int(det[2]*wT), int(det[3]*hT)  #điều chỉnh width & height của box sao cho phù hợp với ảnh
                x,y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2) #điều chỉnh tâm của box
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
                # print(confidence)
                # count += 1
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThresold, nms_threshold) #Loại bỏ các box có tỷ lệ trùng lắm cao hơn mns-threshold
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        color = array.array('i',[random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)])
        cv2.rectangle(img, (x, y), (x+w, y+h), (color[0], color[1], color[2]), 2)
        cv2.putText(img, f'{classNames[classIds[i]]} {int(confs[i]*100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (color[0], color[1], color[2]), 1)

        if confs[i] > 0.3:
            ri += 1
        else:
            wr += 1
    cv2.putText(img, f'Duoc phat hien: {len(indices)}', (10, 13), cv2.FONT_HERSHEY_COMPLEX, 0.6, (48, 48, 255), 2)
    cv2.putText(img, f'Dung: {ri}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (48, 48, 255), 2)
    cv2.putText(img, f'Sai: {wr}', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (48, 48, 255), 2)

blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
net.setInput(blob)

layerNames = net.getLayerNames() #3 layer ['yolo_82', 'yolo_94', 'yolo_106']
# print(layerNames)
# print(net.getUnconnectedOutLayers())

outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
print(outputNames)

outputs = list(net.forward(outputNames))
# print(outputs[0].shape)
# print(outputs[1].shape)
# print(outputs[2].shape)
findOjects(outputs, img)

cv2.imshow('Image', img)
cv2.waitKey()

# while True:
#     success, img = cap.read()
#
#     blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
#     net.setInput(blob)
#
#     layerNames= net.getLayerNames()
#
#     outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
#
#     outputs = list(net.forward(outputNames))
#
#     print(outputs[0].shape)
#     print(net.getUnconnectedOutLayers())
#
#     findOjects(outputs, img)
#
#     cv2.imshow('Image', img)
#     cv2.waitKey(1)