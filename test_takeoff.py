# -*- coding: UTF-8 -*-
#!/usr/bin/python3

#import olympe
#import os
#import time
#from olympe.messages.ardrone3.Piloting import TakeOff, Landing,FlyingStateChanged

#DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
#DRONE_IP = "192.168.42.1"


#if __name__ == "__main__":
 #   drone = olympe.Drone(DRONE_IP)
  #  drone.connect()
   # assert drone(
    #    TakeOff()
        #>>FlyingStateChanged(states="hovering",_timeout=5)
 #   ).wait().success()
  #  time.sleep(10)
   # assert drone(Landing()).wait().success()
    #drone.disconnect()

# -*- coding: UTF-8 -*-
#!/usr/bin/python3
import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, moveBy, Landing
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged

DRONE_IP = os.environ.get("DRONE_IP",192.168.42.1")

if __name__ == "__main__":
    drone = olympe.Drone(DRONE_IP)
    drone.connect()
    assert drone(TakeOff()
        >> FlyingStateChanged(state="hovering", _timeout=10)
    ).wait().success()
    assert drone(moveBy(1, 0, 0, 0)
        >> FlyingStateChanged(state="hovering", _timeout=10)
    ).wait().success()
    assert drone(Landing()).wait().success()
    drone.disconnect()

