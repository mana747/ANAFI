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

# 毎秒0.7mでFm前進後、3秒ポーズ
def go(drone, F):
    print("------------------------------go------------------------------")
    assert drone(
        extended_move_by(F, 0, 0, 0, 0.7, 0.7, 0.7)
    ).wait().success()
    time.sleep(3)

# 毎秒0.2mでFm前進
def forward(drone, F):
    print("------------------------------forward------------------------------")
    assert drone(
        extended_move_by(F, 0, 0, 0, 0.2, 0.2, 0.2)
    ).wait().success()

# 毎秒0.2mでFm前進
def adjustment_right(drone, X):
    print("------------------------------adjustment_right------------------------------")
    assert drone(
        extended_move_by(0, -X, 0, 0, 0.2, 0.2, 0.2)
    ).wait().success()

def adjustment_left(drone, X):
    print("------------------------------adjustment_left------------------------------")
    assert drone(
        extended_move_by(0, X, 0, 0, 0.2, 0.2, 0.2)
    ).wait().success()

# 毎秒0.7mでHm高度上昇
def gain_altitude(drone, H):
    print("------------------------------gain_altitude------------------------------")
    drone(
        extended_move_by(0, 0, -H, 0, 0.7, 0.7, 0.7)
    ).wait().success()
    time.sleep(3)

# GPSを用いて目的地まで毎秒0.7mで移動
def moveto(drone, latitude, longitude):
    print("------------------------------moveto------------------------------")
    drone(
        extended_move_to(latitude, longitude, 3, 0, 0.0, 0.7, 0.7, 0.7)
    ).wait().success()
    time.sleep(3)

# 着陸
def landing(drone):
    print("------------------------------landing------------------------------")
    drone(Landing()).wait().success()
    drone.disconnect()

#
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
        
    """
        for *box, conf, cls in results:
            s = model.names[int(cls)]+":"+'{:.1f}'.format(float(conf)*100)
            print(int(box[0]))
            print(int(box[1]))
            print(int(box[2]))
            print(int(box[3]))

            #--- 枠描画
            cv2.rectangle(
                cv2frame,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color=cc,
                thickness=2,
                )
            #--- 文字枠と文字列描画
            cv2.rectangle(cv2frame, (int(box[0]), int(box[1])-20), (int(box[0])+len(s)*10, int(box[1])), cc, -1)
            cv2.putText(cv2frame, s, (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_PLAIN, 1, cc2, 1, cv2.LINE_AA)


        #--- 描画した画像を表示
        cv2.imshow('color',cv2frame)

    #--- （参考）yolo標準機能を使った出力 ---
    #  results.show()#--- yolo標準の画面表示
    #  results.print()#--- yolo標準のコンソール表示

    #--- （参考）yolo標準の画面を画像取得してopencvで表示 ---
    #  pics = results.render()
    #  pic = pics[0]
    #  cv2.imshow('color',pic)

    #--- 「q」キー操作があればwhileループを抜ける ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        for *box, conf, cls in results:
            if not box:
                print("################################object NOT found##############################################")
            else:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!object found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(int(box[0]))
                print(int(box[1]))
                print(int(box[2]))
                print(int(box[3]))
                

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

        except cv2.error as e:
            print(e)
        except Exception as ne:
            print(ne)
        except KeyboardInterrupt:
            landing(drone)
                      
"""
        #検出結果を取得
        #objects = result.pandas().xyxy[0]
        #print("###################################get result###########################")
        
           
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
        

    
        while True:
            #gain_altitude(drone, 0.5)
            
        
                #移動距離の計算
                PmoveX = centerX - AcenterX
                PmoveY = AcenterY - centerY
                #1pix = 0.00032986[m]
                MmoveX = PmoveX * pix
                MmoveY = PmoveY * pix

                print("#####################移動を開始#####################")
                print(MmoveX + "m左右、" + MmoveY + "m上昇下降します")
                #go_LR(drone, MmoveX)
                #gain_altitude(drone, MmoveY)

                print("#####################接近します#####################")
                #go(drone, 1)

                print("#####################撮影します#####################")

                break
            
        
        #landing(drone)     
        #drone.disconnect()
"""
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

    start_processing = True
    time.sleep(100)

    

  