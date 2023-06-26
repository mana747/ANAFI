import os
from math import *
import olympe
import os

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.move import extended_move_by, extended_move_to
from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged, HomeChanged
from olympe.video.pdraw import Pdraw, PdrawState
from olympe.video.renderer import PdrawRenderer

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

#楕円体
ELLIPSOID_GRS80 = 1 # GRS80
ELLIPSOID_WGS84 = 2 # WGS84

# 楕円体ごとの長軸半径と扁平率
GEODETIC_DATUM = {
    ELLIPSOID_GRS80: [
        6378137.0,         # [GRS80]長軸半径
        1 / 298.257222101, # [GRS80]扁平率
    ],
    ELLIPSOID_WGS84: [
        6378137.0,         # [WGS84]長軸半径
        1 / 298.257223563, # [WGS84]扁平率
    ],
}

# 反復計算の上限回数
ITERATION_LIMIT = 1000

#壁のGPS情報情報 
kabe_lat = 35.710878
kabe_lon = 139.523686

def get_now_gps(drone):
    # Wait for GPS fix
    drone(GPSFixStateChanged(_policy='wait'))
    return drone.get_state(HomeChanged)

def vincenty_inverse(drone,lat,lat0,lon,lon0,ellipsoid=None):
    # 差異が無ければ0を返す
    if abs(lat-lat0)<0.000001 and abs(lon-lon0)<0.000001:
        s, a1, a2=0, 0, 0
        return (s, a1, a2)
    else:
        # 計算時に必要な長軸半径(a)と扁平率(ƒ)を定数から取得し、短軸半径(b)を算出する
        # 楕円体が未指定の場合はGRS80の値を用いる
        a, ƒ = GEODETIC_DATUM.get(ellipsoid, GEODETIC_DATUM.get(ELLIPSOID_GRS80))
        b = (1 - ƒ) * a

        φ1 = radians(lat)
        φ2 = radians(lat0)
        λ1 = radians(lon)
        λ2 = radians(lon0)

        # 更成緯度(補助球上の緯度)
        U1 = atan((1 - ƒ) * tan(φ1))
        U2 = atan((1 - ƒ) * tan(φ2))

        sinU1 = sin(U1)
        sinU2 = sin(U2)
        cosU1 = cos(U1)
        cosU2 = cos(U2)

        # 2点間の経度差
        L = λ2 - λ1

        # λをLで初期化
        λ = L

        # 以下の計算をλが収束するまで反復する
        # 地点によっては収束しないことがあり得るため、反復回数に上限を設ける
        for i in range(ITERATION_LIMIT):
            sinλ = sin(λ)
            cosλ = cos(λ)
            sins = sqrt((cosU2 * sinλ) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosλ) ** 2)
            coss = sinU1 * sinU2 + cosU1 * cosU2 * cosλ
            s = atan2(sins, coss)
            sina = cosU1 * cosU2 * sinλ / sins
            cos2a = 1 - sina ** 2
            cos2sm = coss - 2 * sinU1 * sinU2 / cos2a
            C = ƒ / 16 * cos2a * (4 + ƒ * (4 - 3 * cos2a))
            λʹ = λ
            λ = L + (1 - C) * ƒ * sina * (s + C * sins * (cos2sm + C * coss * (-1 + 2 * cos2sm ** 2)))

            # 偏差が.000000000001以下ならbreak
            if abs(λ - λʹ) <= 1e-12:
                break
        else:
            # 計算が収束しなかった場合はNoneを返す
            return None

        # λが所望の精度まで収束したら以下の計算を行う
        u2 = cos2a * (a ** 2 - b ** 2) / (b ** 2)
        A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
        B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
        Δs = B * sins * (cos2sm + B / 4 * (coss * (-1 + 2 * cos2sm ** 2) - B / 6 * cos2sm * (-3 + 4 * sins ** 2) * (-3 + 4 * cos2sm ** 2)))

        # 2点間の楕円体上の距離
        s = b * A * (s - Δs)

        # 各点における方位角
        a1 = atan2(cosU2 * sinλ, cosU1 * sinU2 - sinU1 * cosU2 * cosλ)
        a2 = atan2(cosU1 * sinλ, -sinU1 * cosU2 + cosU1 * sinU2 * cosλ) + pi

        if a1 < 0:
            a1 = a1 + pi * 2

        return (s, a1, a2)

#距離取得 s:距離 a1:方位角 a2:方位角
def get_distance():
    gps = get_now_gps(drone)
    drone_lat, drone_lon, drone_altitude = gps['latitude'], gps['longitude'], gps['altitude']
    s, a1, a2 = vincenty_inverse(drone, drone_lat, kabe_lat, drone_lon, kabe_lon, ellipsoid=1)

    print("----------------------------------------------distanceYYY-------------------------------------")
    print(drone_lat)
    print(drone_lon)
    print("----------------------------------------------distanceYYY-------------------------------------")

    #テキストファイルに記録
    with open('/home/manaki/code/parrot-groundsdk/experience/GPSdistance/GPSdistance.txt', 'a') as f:
        f.write("\n")
        f.write(str(s))

if __name__ == "__main__":
    # ドローン接続
    drone = olympe.Drone(DRONE_IP)
    drone.connect()
    
    #ドローンGPS取得、距離計算
    get_distance()