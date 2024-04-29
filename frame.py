import cv2

nue = cv2.VideoCapture("nue_ai.mp4")
i = 0
while (nue.isOpened()):
  ret, frame = nue.read()
  cv2.imwrite("nue/nue_" + str(i) + ".jpg", frame)
  i += 1