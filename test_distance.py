from matplotlib.image import imread
import torch
import cv2
import os
import numpy as np
import math
import requests

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.move import extended_move_by, extended_move_to
from olympe.video.pdraw import Pdraw, PdrawState
from olympe.video.renderer import PdrawRenderer
import time
from olympe.messages.camera import (
    set_camera_mode,
    set_photo_mode,
    take_photo,
    photo_progress,
)

# parrot設定
RTSP_URL = 'rtsp://192.168.42.1/live'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
# Drone web server URL
ANAFI_URL = "http://{}/".format(DRONE_IP)
# Drone media web API URL
ANAFI_MEDIA_API_URL = ANAFI_URL + "api/v1/media/medias/"
XMP_TAGS_OF_INTEREST = (
    "CameraRollDegree",
    "CameraPitchDegree",
    "CameraYawDegree",
    "CaptureTsUs",
    # NOTE: GPS metadata is only present if the drone has a GPS fix
    # (i.e. they won't be present indoor)
    "GPSLatitude",
    "GPSLongitude",
    "GPSAltitude",
)

#モデルのロード
model = torch.hub.load('/home/manaki/yolov5','custom',path='/home/manaki/code/parrot-groundsdk/yolo_models/best_peeling2023.pt',source='local')
THICKNESS = 5
COLOR = (255, 0, 0)
COLOR_rectangle = (255, 0, 0)
THICKNESS_rectangle = 5
THICKNESS_text = 5
COLOR_text = (0,255,0)

# 変数の設定
target_found = False
start_processing = False

#ANAFIのピクセル
x = 3840
AcenterX = x/2
y = 2160
AcenterY = y/2

#1pixあたりm
#ピクセルの差分を焦点距離で割ると実際の移動距離を求めらる
#カメラモード、動画、写真によってキャリブレーションをする必要があるかも
pix = 0.00032986

#距離計測時のドローン上昇距離[m]
l = 1

# カメラの内部パラメータ（写真3456:4608）
fx = 2996.78615 #カメラの焦点距離_x[pix/m]
fy = 2992.10096 #カメラの焦点距離_y[pix/m]
cx = 2324.71866 #画像中心のx座標
cy = 1702.21981 #画像中心のy座標

# 離陸
def takeoff(drone):
    print("------------------------------takeoff------------------------------")
    assert drone(TakeOff()).wait().success()
    time.sleep(5)

# 毎秒0.7mでFm前進 前が正
def go(drone, F):
    print("------------------------------go------------------------------")
    assert drone(
        extended_move_by(F, 0, 0, 0, 0.5, 0.5, 0.5)
    ).wait().success()
    time.sleep(3)

# 毎秒0.7mでRm左右進　右が正
def go_LR(drone, R):
    print("------------------------------go------------------------------")
    assert drone(
        extended_move_by(0, R, 0, 0, 0.7, 0.7, 0.7)
    ).wait().success()
    time.sleep(3)

# 毎秒0.7mでHm高度　下が正
def altitude(drone, H):
    print("------------------------------gain_altitude------------------------------")
    drone(
        extended_move_by(0, 0, H, 0, 0.7, 0.7, 0.7)
    ).wait().success()
    time.sleep(3)

# 毎秒0.7mでHm高度上昇　下が正
def gain_altitude(drone, H):
    print("------------------------------gain_altitude------------------------------")
    drone(
        extended_move_by(0, 0, -H, 0, 0.7, 0.7, 0.7)
    ).wait().success()
    time.sleep(3)

# 着陸
def landing(drone):
    print("------------------------------landing------------------------------")
    drone(Landing()).wait().success()
    drone.disconnect()

#カメラモード設定
def setup_photo_burst_mode(drone):
    drone(set_camera_mode(cam_id=0, value="photo")).wait()
    # For the file_format: jpeg is the only available option
    # dng is not supported in burst mode
    drone(
        set_photo_mode(
            cam_id=0,
            mode="single",
            format="rectilinear",
            file_format="jpeg",
            burst="burst_4_over_1s",
            bracketing="preset_1ev",
            capture_interval=0.0,
        )
    ).wait()

#撮影
def take_photo_burst(drone):
    # take a photo burst and get the associated media_id
    photo_saved = drone(photo_progress(result="photo_saved", _policy="wait"))
    drone(take_photo(cam_id=0)).wait()
    photo_saved.wait()
    media_id = photo_saved.received_events().last().args["media_id"]

    return media_id

#写真保存
def save_photo(i):
    media_id = take_photo_burst(drone)
    # Save photo to local storage
    print("*******************************************************************************************************")
    print("photo_" + str(i)+".jpg")
    print("SAVING PICTURE")
    print("*******************************************************************************************************")
    media_info_response = requests.get(ANAFI_MEDIA_API_URL + media_id)
    media_info_response.raise_for_status()
    for resource in media_info_response.json()["resources"]:
        image_response = requests.get(ANAFI_URL + resource["url"])
        open("/home/manaki/code/parrot-groundsdk/DistancePhoto/photo_"+str(i)+".jpg", 'wb').write(image_response.content)

#画像の座標を取得
def Coordinate(i):
    for *box, conf, cls in i.xyxy[0]:
        print(int(box[0]))
        print(int(box[1]))
        print(int(box[2]))
        print(int(box[3]))

        Coordinate = int(box[1])
        return Coordinate

#距離の計算
def cal_d(i, j):
    results1 = model("/home/manaki/code/parrot-groundsdk/DistancePhoto/photo_"+str(i)+".jpg")  # 画像パスを設定し、物体検出を行う
    results2 = model("/home/manaki/code/parrot-groundsdk/DistancePhoto/photo_"+str(j)+".jpg")  # 画像パスを設定し、物体検出を行う

    #座標の取得
    v1 = Coordinate(results1)
    v2 = Coordinate(results2)

    # 画像中心から物体の位置までの垂直方向の距離
    dy1 = v1 - cy #物体１
    dy2 = v2 - cy #物体２

    # 上下方向の角度の計算
    angle_rad1 = np.arctan(dy1 / fy)
    angle_deg1 = np.degrees(angle_rad1)

    angle_rad2 = np.arctan(dy2 / fy)
    angle_deg2 = np.degrees(angle_rad2)

    #絶対値へ変換
    angle_deg1 = abs(angle_deg1)
    angle_deg2 = abs(angle_deg2)

    print("----------------------------------angle_deg-----------------------------------------------")
    print(angle_deg1)
    print(angle_deg2)
    print("----------------------------------angle_deg-----------------------------------------------")

    #tanの計算
    tan1 = math.tan(math.radians(angle_deg1))
    tan2 = math.tan(math.radians(angle_deg2))

    #角度から物体までの距離を計算
    z = l / (tan1 + tan2)

    return z

def frame_processing(yuv_frame):
    if not start_processing:
        print("not start processig")
        return

    print("#####################PROCESSING START#####################")

    global target_found
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
        
        result = None
        for row in results:
            if not row[-1]:
                result=tuple(row.int().numpy()[:-2])
            
        # draw rectangle
        if result is not None:
            start_point = (result[0], result[1])
            end_point = (result[2], result[3])
            cv2frame = cv2.rectangle(cv2frame, result[:2], result[2:], COLOR_rectangle, THICKNESS_rectangle)
            cv2.putText(cv2frame,'Peeling',result[:2],cv2.FONT_HERSHEY_SIMPLEX,1,COLOR_text,THICKNESS_text)
        print("#####################DRAW RECTANGLE#####################")
                
        # show the frame
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", cv2frame)
        # print("#####################FRAME SHOWN#####################")

        #ボックスの座標取得
        #box[0]:左上のX座標　box[1]:左上のY座標 box[2]:右下のX座標 box[3]:右下のY座標
        for *box, conf, cls in results:
            if not box:
                print("################################object NOT found##############################################")
            else:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!object found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                #1枚目の写真保存
                take_photo_burst(drone)
                save_photo(1)

                #理論上は最低でも0.5775[m]上昇する必要がある
                gain_altitude(drone, 1.0)

                #2枚目の写真保存
                take_photo_burst(drone)
                save_photo(2)

                #距離計算
                z = cal_d(1, 2)
                d = z - 1

                with open('10.txt', 'w') as f:
                    print("----------------------------------------------distanceYYY-------------------------------------")
                    print(z, file=f)
                    print("----------------------------------------------distanceYYY-------------------------------------")

                with open('10.txt') as f:
                    print(f.readlines())

                #壁へ接近、撮影
                # go(drone, d)
                # take_photo_burst(drone)
                # save_photo(3)

                target_found = True
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
        # cap.release()
            pass

    except cv2.error as e:
        print(e)
    except Exception as ne:
        print(ne)
    except KeyboardInterrupt:
        landing(drone)
        drone.disconnect()
        
        #landing(drone)     
        #drone.disconnect()

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

    #カメラモードの設定
    setup_photo_burst_mode(drone)

    takeoff(drone)
    gain_altitude(drone, 1)
    start_processing = True

    try:
        while True:
            if target_found:
                start_processing = False
                time.sleep(1)
                landing(drone)
                break
        drone.disconnect()    
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        landing(drone)