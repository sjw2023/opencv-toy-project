import math
import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)

print('width : %d, height: %d' %(cap.get(3), cap.get(4)))


# model
model = YOLO("yolo-Weights/yolov8n.pt")

# Train a model
results = model.train(data=" /Users/joowon/Documents/GitHub/opencv-toy-project/Logistics.v10i.yolov8/data.yaml")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
def markBox( img ):
    # Load the image
    # img = cv2.imread('box.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(img, 19, 215)
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 3, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    thresh = cv2.dilate(thresh,None,iterations =1)
    thresh = cv2.erode(thresh,None,iterations =1)

    # Find the contours
    contours,hierarchy = cv2.findContours(thresh,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, find the bounding rectangle and draw it
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >33000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img,
                        (x,y),(x+w,y+h),
                        (0,255,0),
                        5)

    cv2.imshow('img',img)
def markBox2(img):
        # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(img, 9, 99)
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 3, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    thresh = cv2.dilate(thresh,None,iterations =1)
    thresh = cv2.erode(thresh,None,iterations =1)

    # Find the contours
    contours,hierarchy = cv2.findContours(thresh,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, find the bounding rectangle and draw it
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >33000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img,
                        (x,y),(x+w,y+h),
                        (0,255,0),
                        5)

def basicScreenPrint(frame):
    if(ret):
        gray = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)

        cv2.imshow('frame_color', frame)
        cv2.imshow('frame_gray', gray)
def yoloClassNames():
    print( model.names )
def yoloRecognition(frame):
    results = model(frame, stream=True)
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)
            
while(True):
    ret, frame = cap.read()
    
    # markBox2(frame)
    yoloClassNames()
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





