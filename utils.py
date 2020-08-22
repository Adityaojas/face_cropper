import numpy as np
import imutils
import cv2

def crop_face(in_path, out_path, mod_path, prtxt_path, square_side):
    net = cv2.dnn.readNetFromCaffe(prtxt_path, mod_path)
    cap = cv2.VideoCapture(in_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    #cap.release()
    
    if cap.isOpened == False:
        print('Error opening the file')
    
    output = cv2.VideoWriter(out_path, 0, fps, (square_side, square_side))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    (f_h, f_w) = (square_side, square_side)
    zeros = None
        
    while True:
        (grabbed, frame) = cap.read()
        frame_copy = frame.copy()
        #frame = imutils.resize(frame, width = 400)
        
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            midX = (startX+endX) // 2
            midY = (startY+endY) // 2
            
            x1 = midX - square_side//2
            x2 = midX + square_side//2
            y1 = midY - square_side//2
            y2 = midY + square_side//2
            
            #text = "{:.2f}%".format(confidence * 100)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            #cv2.rectangle(frame, (x1, y1), (x2, y2),
            #              (0, 0, 255), 2)
            #cv2.putText(frame, text, (startX, y),
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        if writer is None:
            writer = cv2.VideoWriter(out_path, fourcc, fps,
                                     (f_w, f_h), True)
            zeros = np.zeros((f_h, f_w), dtype="uint8")
        
        writer.write(frame_copy[y1:y2, x1:x2])
        
        #cv2.imshow("Frame", frame)
        #key = cv2.waitKey(1) & 0xFF
        
        #if key == ord("q"):
        #   break
    
    cap.release()
    #cv2.destroyAllWindows()
