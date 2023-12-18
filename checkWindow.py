from matplotlib.image import imread
import torch
import cv2
import os
import numpy as np
import math
import requests
import itertools

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
model = torch.hub.load('/home/manaki/yolov5','custom',path='/home/manaki/code/parrot-groundsdk/yolo_models/best_tamawin0816.pt',source='local') #特徴的な構造物から損傷予測モデル
COLOR = (255, 0, 0)
COLOR_rectangle = (255, 0, 0)
THICKNESS_rectangle = 5
THICKNESS_text = 5
COLOR_text = (0,255,0)

# クラスラベルの定義
class_labels = ['window', 'ExhaustPort', 'door']  # 検出したいクラスのラベルを定義

# 変数の設定
target_found = False
start_processing = False

#ANAFIのピクセル
x = 4608
AcenterX = x/2
y = 3456
AcenterY = y/2

#cv2frame to anafipixスケーリングファクター
height_ratio = 3456 / 720
width_ratio = 4608 / 1280

#距離計算時の条件
CONFIDENCE_THRESHOLD = 0.5
FRAME_WIDTH = 4608
FRAME_HEIGHT = 3456

#1pixあたりm
pix = 0.00032986

#距離計測時のドローン上昇距離[m]
l = 1

# カメラの内部パラメータ（写真3456:4608）
fx = 2996.78615 #カメラの焦点距離_x[pix/m]
fy = 2992.10096 #カメラの焦点距離_y[pix/m]
cx = 2324.71866 #画像中心のx座標
cy = 1702.21981 #画像中心のy座標

# カメラ行列
camera_matrix = np.array([[2.99678615e+03, 0.00000000e+00, 2.32471866e+03],
                          [0.00000000e+00, 2.99210096e+03, 1.70221981e+03],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# 歪み係数
distortion_coeffs = np.array([0.00078086, 0.01357406, -0.00225809, 0.00225071, -0.0249281])

#メモ
#カメラフレームの大きさ
#写真3456:4608、動画2160:3840

#動作前の許可確認
def ask_permission():
    while True:
        response = input("続行しますか？(Y/N): ").strip().lower()
        if response == 'y':
            return True
        elif response == 'n':
            return False
        else:
            print("無効な入力です。YかNで答えてください。")

# 歪み補正の関数
def undistort_image(image, camera_matrix, distortion_coeffs):
    h, w = image.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_coeffs, None, new_camera_matrix)
    return undistorted_image

# 離陸
def takeoff(drone):
    print("------------------------------takeoff------------------------------")
    assert drone(TakeOff()).wait().success()
    time.sleep(5)

# 毎秒0.7mでFm前進 前が正
def go(drone, F):
    print("------------------------------go------------------------------")
    assert drone(
        extended_move_by(F, 0, 0, 0, 0.7, 0.7, 0.7)
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
file_counter = 1  # グローバル変数でファイル名を管理する
def save_photo():
    global file_counter  # グローバル変数を使用するために宣言
    media_id = take_photo_burst(drone)
    # Save photo to local storage
    print("*******************************************************************************************************")
    print("photo_" + str(file_counter)+".jpg")
    print("SAVING PICTURE")
    print("*******************************************************************************************************")
    media_info_response = requests.get(ANAFI_MEDIA_API_URL + media_id)
    media_info_response.raise_for_status()
    for resource in media_info_response.json()["resources"]:
        image_response = requests.get(ANAFI_URL + resource["url"])
        open("/home/manaki/code/parrot-groundsdk/DistancePhoto/photo_"+str(file_counter)+".jpg", 'wb').write(image_response.content)
    file_counter += 1

#距離の計算
def cal_d(model):
    print("caldistance")
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    # zを初期化
    z = None

    # 1枚目の写真の検出結果を保存する変数
    detected_object1 = None
    cald_y1 = None
    cald_y2 = None

    # 1枚目の写真保存
    take_photo_burst(drone)
    save_photo()
    time.sleep(3)

    # 歪みのある画像の読み込み
    distorted_image = cv2.imread("/home/manaki/code/parrot-groundsdk/DistancePhoto/photo_" + str(file_counter - 1 ) + ".jpg")

    # 歪みを補正した画像を取得
    undistorted_image = undistort_image(distorted_image, camera_matrix, distortion_coeffs)

    # 歪み補正した画像を保存
    cv2.imwrite("/home/manaki/code/parrot-groundsdk/DistancePhoto/undistorted_image1.jpg", undistorted_image)

    # 1枚目の写真で物体検出を行う
    results1 = model("/home/manaki/code/parrot-groundsdk/DistancePhoto/undistorted_image1.jpg")

    # result1 の判定
    if results1.pred[0].shape[0] > 0:
        print("物体が検出されました。")
        pred_boxes1 = results1.pred[0].detach().cpu().numpy()[:, :4]
        pred_classes1 = results1.pred[0].detach().cpu().numpy()[:, 5].astype(int)
        pred_scores1 = results1.pred[0].detach().cpu().numpy()[:, 4]

        for idx, box in enumerate(pred_boxes1):
            x1, y1, x2, y2 = box
            class_label = class_labels[pred_classes1[idx]]
            confidence = pred_scores1[idx]
            print(f'{class_label} detected at ({x1}, {y1}) - ({x2}, {y2}), confidence: {confidence}')

            # 物体の検出結果が信頼性高く、フレーム内にあり、y1が上半分にある場合に保存
            if confidence > CONFIDENCE_THRESHOLD and 0 <= x1 <= x2 <= FRAME_WIDTH and 0 <= y1 <= FRAME_HEIGHT // 2:
                detected_object1 = x1, x2
                detected_label1 = class_label
                cald_y1 = y1

                # 画像の読み込み
                image_path = "/home/manaki/code/parrot-groundsdk/DistancePhoto/undistorted_image1.jpg"
                image = cv2.imread(image_path)

                # ボックスの座標 (x_min, y_min, x_max, y_max)
                print(x1)
                print(y1)
                print(x2)
                print(y2)
                box_coords = (x1, y1, x2, y2)
                print(box)

                # ボックスを描画
                color = (0, 255, 0)  # 色は緑 (BGR形式)
                thickness = 2  # 線の太さ
                cv2.rectangle(image, (int(box_coords[0]), int(box_coords[1])), (int(box_coords[2]), int(box_coords[3])), color, thickness)

                # 描画した画像を保存
                output_path = "/home/manaki/code/parrot-groundsdk/DistancePhoto/draw_3.jpg"
                cv2.imwrite(output_path, image)

                print(f"Image with box saved as '{output_path}'")

                break

    #理論上は最低でも0.5775[m]上昇する必要がある
    gain_altitude(drone, 1.0)

    # 2枚目の写真保存
    take_photo_burst(drone)
    save_photo()
    time.sleep(3)

    #1m下降し、もとの位置へ
    altitude(drone, 1.0)

    # 歪みのある画像の読み込み
    distorted_image = cv2.imread("/home/manaki/code/parrot-groundsdk/DistancePhoto/photo_" + str(file_counter - 1) + ".jpg")

    # 歪みを補正した画像を取得
    undistorted_image = undistort_image(distorted_image, camera_matrix, distortion_coeffs)

    # 歪み補正した画像を保存
    cv2.imwrite("/home/manaki/code/parrot-groundsdk/DistancePhoto/undistorted_image2.jpg", undistorted_image)

    # 2枚目の写真で物体検出を行う
    results2 = model("/home/manaki/code/parrot-groundsdk/DistancePhoto/undistorted_image2.jpg")

    # 2枚目の写真で同じ物体が再度検出されるかを確認し、条件を満たす場合にのみ物体までの距離を推定
    if detected_object1 is not None:
        for idx, box in enumerate(results2.pred[0].detach().cpu().numpy()[:, :4]):
            x1, y1, x2, y2 = box
            cald_y2 = y1
            class_label2 = class_labels[pred_classes1[idx]]
            print(f'{class_label2} detected at ({x1}, {y1}) - ({x2}, {y2}), confidence: {confidence}')

            # detected_objectと同じ物体を再度検出し、条件を満たす場合にのみ物体までの距離を推定
            if detected_label1 == class_label2 and np.allclose(detected_object1, [x1, x2], rtol=0, atol=10000):  # 位置が近似しているかを確認するために誤差を許容
                print("同じ物体が再度検出されました。")

                # 画像の読み込み
                image_path = "/home/manaki/code/parrot-groundsdk/DistancePhoto/undistorted_image2.jpg"
                image = cv2.imread(image_path)

                # ボックスの座標 (x_min, y_min, x_max, y_max)
                box_coords = (x1, y1, x2, y2)

                # ボックスを描画
                color = (0, 255, 0)  # 色は緑 (BGR形式)
                thickness = 2  # 線の太さ
                cv2.rectangle(image, (int(box_coords[0]), int(box_coords[1])), (int(box_coords[2]), int(box_coords[3])), color, thickness)

                # 描画した画像を保存
                output_path = "/home/manaki/code/parrot-groundsdk/DistancePhoto/draw_4.jpg"
                cv2.imwrite(output_path, image)

                print(f"Image with box saved as '{output_path}'")

                # 物体までの距離を推定する処理
                # 画像中心から物体の位置までの垂直方向の距離
                dy1 = cald_y1 - cy #物体１
                dy2 = cald_y2 - cy #物体２

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
                
            else:
                print("物体が再度検出されませんでした。")
                landing(drone)
                drone.disconnect()
    else:
        print("物体が検出されませんでした。")
        landing(drone)
        drone.disconnect()

    return z

def check_window(keiro, move, d):
    try:
        for i in range(len(keiro)):
            x1 = keiro[i][0].item()
            y1 = keiro[i][1].item()
            x2 = keiro[i][2].item()
            y2 = keiro[i][3].item()

            #写真フレームサイズへ変換
            x1 = x1 * width_ratio
            y1 = y1 * height_ratio
            x2 = x2 * width_ratio
            y2 = y2 * height_ratio

            if i < len(move):
                xmove = move[i][0].item()
                ymove = move[i][1].item()

            if i == 0:
                #移動距離の計算(pix)
                dx = x1 - AcenterX
                dy = y1 - AcenterY

                # 上下方向の角度の計算
                angle_rad1 = np.arctan(dx / fx)
                angle_deg1 = np.degrees(angle_rad1)

                angle_rad2 = np.arctan(dy / fy)
                angle_deg2 = np.degrees(angle_rad2)

                #絶対値へ変換
                angle_degX = abs(angle_deg1)
                angle_degY = abs(angle_deg2)

                #移動距離の計算
                D = d - 1
                
                if dx > 0:
                    x = d * math.tan(math.radians(angle_degX))
                else:
                    x = -d * math.tan(math.radians(angle_degX))

                if dy > 0:
                    y = d * math.tan(math.radians(angle_degY))
                else:
                    y = -d * math.tan(math.radians(angle_degY))

            #対象物の縦と横の長さを計算
            # バウンディングボックスのピクセル単位の幅と高さを計算
            width_pixels = x2 - x1
            height_pixels = y2 - y1

            # 物体の実際の高さと幅を計算
            width_real = (width_pixels * d) / fx
            height_real = (height_pixels * d) / fy

            #計算結果の表示
            if i == 0:
                print("x方向の移動距離は" + str(x) + "です。y方向の移動距離は" + str(y) + "です。前進距離は" + str(D) + "です。")

            print("また、対象物の高さは" + str(height_real) + "です。幅は" + str(width_real) + "です。")

            #移動と撮影
            if ask_permission():
                print("続行します")
                if i == 0:
                    go_LR(drone, x)
                    altitude(drone, y)
                    go(drone, D)

                take_photo_burst(drone)
                save_photo()

                go_LR(drone, width_real)
                take_photo_burst(drone)
                save_photo()

                altitude(drone, height_real)
                take_photo_burst(drone)
                save_photo()

                go_LR(drone, -width_real)
                take_photo_burst(drone)
                save_photo()

                gain_altitude(drone, height_real)

            else:
                print("キャンセルされました")
                landing(drone)
                drone.disconnect()

            if i < len(move):
                #計算結果の表示
                print("次の窓へ移動します。x方向の移動距離は" + str(xmove) + "です。y方向の移動距離は" + str(ymove) + "です。")
                if ask_permission():
                    print("続行します")
                    #移動と撮影
                    go_LR(drone, xmove)
                    altitude(drone, ymove)

                else:
                    print("キャンセルされました")
                    landing(drone)
                    drone.disconnect()

    except cv2.error as e:
        print(e)
    except Exception as ne:
        print(ne)
    except KeyboardInterrupt:
        landing(drone)
        drone.disconnect()

def calculate_distance(p1, p2):
    # 2つの座標点p1(x1, y1)とp2(x2, y2)の距離を計算する関数
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def total_distance(route):
    # 全ての点を通る経路routeの総移動距離を計算する関数
    distance = 0
    for i in range(len(route) - 1):
        distance += calculate_distance(route[i], route[i + 1])
    distance += calculate_distance(route[-1], route[0])  # 最後の点と最初の点の距離を加算
    return distance

def all_permutations(points):
    # 全ての組み合わせを試す全探索を行う関数
    min_distance = float('inf')
    optimal_route = None

    # 空のリストをチェック
    if not points:
        return optimal_route, min_distance

    for perm in itertools.permutations(points):
        distance = total_distance(perm)
        if distance < min_distance:
            min_distance = distance
            optimal_route = perm

    return optimal_route, min_distance

#経路間の実距離を計算
def cal_saiteki_move(s1, s2, d):
    #ピクセルでの移動距離
    x_pix = s2[0] - s1[0]
    y_pix = s2[1] - s1[1]

    #写真フレームサイズへ変換
    x_pix = x_pix * width_ratio
    y_pix = y_pix * height_ratio

    # 物体の実際の高さと幅を計算
    saitekimoveX = (x_pix * d) / fx
    saitekimoveY = (y_pix * d) / fy

    return [saitekimoveX, saitekimoveY]

#遠距離からの損傷、特徴調査、距離取得
def inspection_damage(model, frame, d):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!START INSPECTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # 物体の座標を格納するリストを初期化
    detected_objects = []

    #物体検出
    results = model([frame]).xyxy[0]

    # 物体が検出されたかどうかを判断
    if len(results.red) > 0:
        detections = results.pred[0]  # Detectionsオブジェクトを取得
        for i in range(len(detections)):
            row = detections[i]
            if row[4] != 0:
                # 物体の座標をリストに追加
                detected_objects.append((row[0], row[1], row[2], row[3]))  # 4つの値を持つタプルとして追加
                print(detected_objects)
   
    else:
        print('No objects detected.')

    # 物体検出で得られた座標から左上の座標のみを抽出
    left_upper_coords = [(x_min, y_min) for x_min, y_min, _, _ in detected_objects]

    # 左上の座標を使って全探索を行う
    optimal_route, min_distance = all_permutations(left_upper_coords)
    if optimal_route is None:
        print("最適経路が見つかりませんでした。")
    else:
        print("最適経路の左上座標:", optimal_route)
        print("総移動距離:", min_distance)

    # optimal_routeに格納されている左上の座標の順番に対応する物体の座標を取得し、saiteki_keiroに追加する
    saiteki_keiro = []
    for left_upper_coord in optimal_route:
        for obj_left_upper_coord in detected_objects:
            obj_x_min, obj_y_min, _, _ = obj_left_upper_coord
            if left_upper_coord == (obj_x_min.item(), obj_y_min.item()):
                saiteki_keiro.append(obj_left_upper_coord)

    #物体間の移動量を計算する関数へ
    saitekimove_list = []
    for i in range(len(optimal_route) - 1):
        result = cal_saiteki_move(optimal_route[i], optimal_route[i+1], d)
        saitekimove_list.append(result)
    
    return saiteki_keiro, saitekimove_list
        
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

        #壁までの距離計算
        #d = cal_d(model)
        d = 5

        #損傷有無のチェックと最短経路探索
        saiteki_keiro, saitekimove_list = inspection_damage(model, cv2frame, d)

        #窓調査
        check_window(saiteki_keiro, saitekimove_list, d)

        # print("")
        # if ask_permission():
        #     print("続行します")
        #     gain_altitude(drone, 1)
        
        # else:
        #     print("キャンセルされました")
        #     landing(drone)
        #     drone.disconnect()

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

if __name__ == '__main__':
    # ドローン接続
    drone = olympe.Drone(DRONE_IP)
    drone.connect()

    #撮影準備
    setup_photo_burst_mode(drone)

    # カメラ起動
    pdraw = Pdraw()
    pdraw.set_callbacks(raw_cb=frame_processing)
    pdraw.play(url=RTSP_URL)
    #renderer = PdrawRenderer(pdraw=pdraw)

    assert pdraw.wait(PdrawState.Playing, timeout=5)
    print("#####################開始#####################")

    takeoff(drone)
    time.sleep(1)
    #gain_altitude(drone, 1)
    #time.sleep(3)
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