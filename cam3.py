import cv2
# import torch
from ultralytics import YOLO
# import time

# print(torch.cuda.is_available())

# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
RED = (0,0,255)
video_cap = cv2.VideoCapture('py\output.mp4')
CLASS=["green_buoy","red_buoy"]
model = YOLO("py/yolov8nano_100_epochs.pt")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('inference_result.mp4', 
                      fourcc=fourcc,
                      frameSize=(640, 480),
                      fps=30.0)


while video_cap.isOpened():
    success, frame = video_cap.read()
    max_green_area=0
    max_red_area=0
    bgreen_tensor=None
    bred_tensor=None
    
    if success:
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD)
        # an=results[0].plot()
        for box in results[0].boxes:
            w=box.xywh[0][2]
            h=box.xywh[0][3]
            area=max(w*w,h*h)
            
            if CLASS[int(box.cls)]=="green_buoy":
                print("ini warna hijau")
                if max_green_area<area:
                    max_green_area=area
                    bgreen_tensor=box
                     
            elif CLASS[int(box.cls)]=="red_buoy":
                print("ini warna merah")
                if max_red_area<area:
                    max_red_area=area
                    bred_tensor=box
            
        if(bgreen_tensor!=None):
            x1_green,y1_green,x2_green,y2_green=bgreen_tensor.xyxy[0]
            x1_green,y1_green,x2_green,y2_green = map(int, [x1_green, y1_green, x2_green, y2_green])
            cv2.rectangle(frame, (x1_green, y1_green), (x2_green, y2_green), GREEN, 2) 
            
        if(bred_tensor!=None):
            x1_red,y1_red,x2_red,y2_red = bred_tensor.xyxy[0]
            x1_red,y1_red,x2_red,y2_red = map(int, [x1_red,y1_red,x2_red,y2_red])
            cv2.rectangle(frame, (x1_red, y1_red), (x2_red, y2_red), RED, 2)

        if bgreen_tensor and bred_tensor:
            bottom_x_y = (round(640/2), 480) # Based on img size e.g 640x480
            # Create line between two bounding box
            cv2.line(frame, (x2_green, y1_green), (x1_red, y1_red), color=(225, 0, 0), thickness=2)
            line_x_top_center = round((x1_green+x2_red)/2)  # Get line's x top position by taking average of x
            line_y_top_center = round((y1_green+y1_red)/2) # Get line's y top position by taking average of y
            line_center_x_y = (line_x_top_center, line_y_top_center)
            # Create line from bottom of picture to middle of line
            cv2.line(frame, bottom_x_y, line_center_x_y, thickness=2, color=(255,254,254))
        # out.write(frame) 
            
        cv2.imshow("yolo",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
video_cap.release() 
# out.release()
cv2.destroyAllWindows()
