import cv2
import datetime
from ultralytics.ultralytics import YOLO

# カメラの起動(0は内蔵カメラが着いている場合は内蔵カメラが起動)
cap = cv2.VideoCapture(0)

while(True):
    # 現在の画像の読み込み
    ret, frame = cap.read()
    # 読み込んだ画像を表示する
    cv2.imshow("frame", frame)
    # qが押されたら止める
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # 現時点での画像を保存する
        time = datetime.datetime.now()
        timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
        #print(str(timestr) + ".jpg")
        cv2.imwrite('hozon/'+str(timestr) + ".jpg", frame)
    if 0xff==ord('w'):
        break
    model = YOLO('./runs/detect/train10/weights/last.pt')