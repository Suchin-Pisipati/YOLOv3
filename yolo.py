import cv2  
import numpy as np  

# Load YOLO  
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')  
layer_names = net.getLayerNames()  
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  

# Load video  
cap = cv2.VideoCapture(0)  

while True:  
    # Capture frame-by-frame  
    _, frame = cap.read()  
    
    # Prepare the frame for the model  
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  
    net.setInput(blob)  
    outs = net.forward(output_layers)  
    
    # Process the outputs  
    for out in outs:  
        for detection in out:  
            scores = detection[5:]  
            class_id = np.argmax(scores)  
            confidence = scores[class_id]  
            if confidence > 0.5:  # Confidence threshold  
                # Object detected  
                center_x = int(detection[0] * frame.shape[1])  
                center_y = int(detection[1] * frame.shape[0])  
                w = int(detection[2] * frame.shape[1])  
                h = int(detection[3] * frame.shape[0])  
                
                # Rectangle coordinates  
                x = int(center_x - w / 2)  
                y = int(center_y - h / 2)  
                
                # Draw the bounding box  
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
                cv2.putText(frame, str(class_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  
    
    # Display the frame with detections  
    cv2.imshow('Image', frame)  

    # Break loop on ESC key  
    if cv2.waitKey(1) & 0xFF == 27:  
        break  

# Release VideoCapture and destroy windows  
cap.release()  
cv2.destroyAllWindows() 