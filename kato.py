import cv2
from cv2 import aruco

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.move import extended_move_by, extended_move_to
import time
import os
import numpy as np
from olympe.video.pdraw import Pdraw, PdrawState
from olympe.video.renderer import PdrawRenderer

# parrot設定
RTSP_URL = 'rtsp://192.168.42.1/live'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")

# aruco設定
dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

# 変数の設定
target_found = False
start_processing = False

# 法政大学小金井キャンパスの中庭、白い四角タイルの角
agri_latitude = 35.709750
agri_longitude = 139.523336

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
def frame_processing(frame):
    if not start_processing:
        return
    global target_found
    try:
        # convert frame to rgb
        cv2_convert_flag = {
            olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12
        }[frame.format()]
        gray = cv2.cvtColor(frame.as_ndarray(), cv2_convert_flag)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(frame.as_ndarray(), corners, ids)
        cv2.imshow('frame', frame_markers)

        # idをリストで表示
        list_ids = list(np.ravel(ids))
        list_ids.sort()
        print(list_ids)

        # マーカーが見つかるまで前進
        forward(drone, 0.4)
        time.sleep(0.05)
        if list_ids[0] == 0:
            print("*********************************************************************")
            print("***************************landing posture***************************")
            print("*********************************************************************")

            index = np.where(ids == 0)[0][0]  # num_id が格納されているindexを抽出
            cornerUL = corners[index][0][0]
            cornerUR = corners[index][0][1]
            cornerBR = corners[index][0][2]
            cornerBL = corners[index][0][3]

            center = [(cornerUL[0] + cornerBR[0]) / 2, (cornerUL[1] + cornerBR[1]) / 2] # マーカー中心の計算

            # print('左上 : {}'.format(cornerUL))
            # print('右上 : {}'.format(cornerUR))
            # print('右下 : {}'.format(cornerBR))
            # print('左下 : {}'.format(cornerBL))
            print('中心 : {}'.format(center))

            if center[1] >= 400 and 570 < center[0] < 630:
                print("*********************************************************************")
                print("*********************************landing*****************************")
                print("*********************************************************************")
                target_found = True
            # 横方向微調整
            elif center[0] < 580:
                adjustment_right(drone, 0.2)
                time.sleep(2)
            elif center[0] > 620:
                adjustment_left(drone, 0.2)
                time.sleep(2)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cap.release()
            pass
    except cv2.error as e:
        print(e)
    except Exception as ne:
        print(ne)
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

    assert pdraw.wait(PdrawState.Playing, timeout=5)
    print("**********************start**********************")

    takeoff(drone)
    time.sleep(1)
    gain_altitude(drone, 2)
    time.sleep(1)
    moveto(drone, agri_latitude, agri_longitude)
    time.sleep(20)
    start_processing = True
    try:
        while True:
            if target_found:
                start_processing = False
                time.sleep(1)
                go(drone, 1.5)
                time.sleep(1)
                landing(drone)
                break
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        landing(drone)
