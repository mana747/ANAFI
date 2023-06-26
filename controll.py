from matplotlib.image import imread
import torch
import cv2
import os
import numpy as np

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.move import extended_move_by, extended_move_to
from olympe.video.pdraw import Pdraw, PdrawState
from olympe.video.renderer import PdrawRenderer
import time

# parrot設定
RTSP_URL = 'rtsp://192.168.42.1/live'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")

#モデルのロード
model = torch.hub.load('/home/manaki/yolov5','custom',path='/home/manaki/code/parrot-groundsdk/yolo_models/best_peeling2023.pt',source='local')
THICKNESS = 5
COLOR = (255, 0, 0)
print('2')

# 変数の設定
target_found = False
start_processing = False

#ANAFIのピクセル
x = 3840
AcenterX = x/2
y = 2160
AcenterY = y/2

#1pixあたりm
pix = 0.00032986

# 離陸
def takeoff(drone):
    print("------------------------------takeoff------------------------------")
    assert drone(TakeOff()).wait().success()
    time.sleep(5)

# 毎秒0.7mでFm前進
def go(drone, F):
    print("------------------------------go------------------------------")
    assert drone(
        extended_move_by(F, 0, 0, 0, 0.7, 0.7, 0.7)
    ).wait().success()
    time.sleep(3)

# 毎秒0.7mでRm左右進
def go_LR(drone, R):
    print("------------------------------go------------------------------")
    assert drone(
        extended_move_by(0, R, 0, 0, 0.7, 0.7, 0.7)
    ).wait().success()
    time.sleep(3)

# 毎秒0.7mでHm高度
def gain_altitude(drone, H):
    print("------------------------------gain_altitude------------------------------")
    drone(
        extended_move_by(0, 0, H, 0, 0.7, 0.7, 0.7)
    ).wait().success()
    time.sleep(3)

# 着陸
def landing(drone):
    print("------------------------------landing------------------------------")
    drone(Landing()).wait().success()
    drone.disconnect()

def frame_processing(yuv_frame):
    if not start_processing:
        print("not start processig")
        return

    print("#####################PROCESSING START#####################")

    try:
        info = yuv_frame.info()
        
        height, width = (  # noqa
            info["raw"]["frame"]["info"]["height"],
            info["raw"]["frame"]["info"]["width"],
        )
        
        cv2_cvt_color_flag = {
            olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
        }[yuv_frame.format()]
        
        #pdraw YUV から Opencv YUV へ変換
        cv2frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)
      	
        #物体検出
        results = model([cv2frame]).xyxy[0]
        #検出結果を取得
        objects = results.pandas().xyxy[0]

        result = None
        for row in results:
            if not row[-1]:
                result=tuple(row.int().numpy()[:-2])
        
        # draw rectangle
        if result is not None:
            start_point = (result[0], result[1])
            end_point = (result[2], result[3])
            cv2frame = cv2.rectangle(cv2frame, result[:2], result[2:], COLOR, THICKNESS)
            cv2.putText(cv2frame,'Peeling',result[:2],cv2.FONT_HERSHEY_SIMPLEX,1,COLOR_text,THICKNESS_text)
        print("#####################DRAW RECTANGLE#####################")
               
        # show the frame
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", cv2frame)
        print("#####################FRAME SHOWN#####################")
        
        while True:
            #gain_altitude(drone, 0.5)
            if len(objects) != 0:
                #情報の取得
                for i in range(len(objects)):
                    name = objects.name[i]
                    xmin = objects.xmin[i]
                    ymin = objects.ymin[i]
                    width = objects.xmax[i] - objects.xmin[i]
                    height = objects.ymax[i] - objects.ymin[i]
                    centerX = xmin + width/2
                    centerY = ymin + height/2
                    print("中心のＸ座標")
                    print(centerX)
                    print("中心のY座標")
                    print(centerY)
        
                #移動距離の計算
                PmoveX = centerX - AcenterX
                PmoveY = AcenterY - centerY
                #1pix = 0.00032986[m]
                MmoveX = PmoveX * pix
                MmoveY = PmoveY * pix

                print("#####################移動を開始#####################")
                print(MmoveX + "m左右、" + MmoveY + "m上昇下降します")
                go_LR(drone, MmoveX)
                gain_altitude(drone, MmoveY)

                print("#####################接近します#####################")
                #go(drone, 1)

                print("#####################撮影します#####################")

                break

            elif len(objects) == 0:
                pass

        landing(drone)     
        drone.disconnect()
           
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cap.release()
            pass
    except KeyboardInterrupt:
        landing(drone)

if __name__ == '__main__':
    # ドローン接続
    drone = olympe.Drone(DRONE_IP)
    drone.connect()

    # カメラ起動
    pdraw = Pdraw()
    pdraw.set_callbacks(raw_cb=frame_processing)
    pdraw.play(url=RTSP_URL)
    #renderer = PdrawRenderer(pdraw=pdraw)

    assert pdraw.wait(PdrawState.Playing, timeout=5)
    print("#####################開始#####################")
    
    takeoff(drone)
    time.sleep(1)
    gain_altitude(drone, 1)
    time.sleep(1)
    start_processing = True
