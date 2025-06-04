import copy
import threading
import time
from pyzed import sl

import Settings as s
from Joint import Joint
from PyZedWrapper import PyZedWrapper


class DistShouldersMeasurment(threading.Thread):
    def __init__(self,):
        threading.Thread.__init__(self)

        # Initialize the ZED camera
        print("CAMERA INITIALIZATION")
        self.dist_list = []  # Store joints data

    def run(self):
        print("CAMERA START")
        zed = PyZedWrapper()
        zed.start()

        for i in range (0,20): #from 0 to 19
            self.get_skeleton_data()

        average = sum(self.dist_list) / len(self.dist_list)
        print(str(average))


    def get_skeleton_data(self):
        bodies = sl.Bodies()  # Structure containing all the detected bodies
        body_runtime_param = sl.BodyTrackingRuntimeParameters()
        body_runtime_param.detection_confidence_threshold = 40

        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_bodies(bodies, body_runtime_param)
            body_array = bodies.body_list
            num_of_bodies = len(body_array)
            time.sleep(0.001)

            if num_of_bodies != 0:
                body = bodies.body_list[0]

                for i, kp_3d in enumerate(body.keypoint):
                    if i==12:
                        l_shoulder_x = kp_3d[0]

                    if i==13:
                        r_shoulder_x = kp_3d[0]


                self.dist_list.append(copy.deepcopy(abs(l_shoulder_x-r_shoulder_x)))



            else:
                return None

        else:
            return None

if __name__ == "__main__":
    # Ensure the script is being run directly
    print("Starting Shoulder Distance Measurement")
    time.sleep(7)
    dist_measurement = DistShouldersMeasurment()
    dist_measurement.start()  # Start the thread

    # Wait for the thread to complete
    dist_measurement.join()
    print("Measurement completed.")
