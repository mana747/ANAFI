import olympe
import os
from olympe.messages.ardrone3.PilotingSettingsState import MaxTiltChanged

DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")


def test_maxtiltget():
    drone = olympe.Drone(DRONE_IP)
    drone.connect()
    print("Drone MaxTilt = ", drone.get_state(MaxTiltChanged)["current"])
    drone.disconnect()


if __name__ == "__main__":
    test_maxtiltget()
