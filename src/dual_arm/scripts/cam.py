import cv2
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

print("Camera FPS:", fps)