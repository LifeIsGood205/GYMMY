
import time
import pyzed.sl as sl
import threading
import socket
from Audio import say, get_wav_duration
from Joint_zed import Joint
from PyZedWrapper import PyZedWrapper
import Settings as s
import numpy as np


class Help (threading.Thread):

    def calc_angle_3d(self, joint1, joint2, joint3):
        a = np.array([joint1.x, joint1.y, joint1.z], dtype=np.float32)  # First
        b = np.array([joint2.x, joint2.y, joint2.z], dtype=np.float32)  # Mid
        c = np.array([joint3.x, joint3.y, joint3.z], dtype=np.float32)  # End

        ba = a - b
        bc = c - b
        try:
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            return round(np.degrees(angle), 2)

        except:
            print("could not calculate the angle")
            return None

    def _init_(self):
        threading.Thread._init_(self)
        # Initialize the ZED camera
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = ('localhost', 7000)
        self.sock.bind(self.server_address)
        print("CAMERA INITIALIZATION")
        self.frame_count = 0
        self.start_time = None

    def run(self):
        while True:
            print("CAMERA START")
            medaip = PyZedWrapper()
            medaip.start()
            self.zed = PyZedWrapper.get_zed(medaip)

            while not s.finish_program:
                self.get_skeleton_data()

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

                arr_organs = ['pelvis', 'spine_1', 'spine_2', 'spine_3', 'neck', 'nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
                              'L_clavicle', 'R_clavicle', 'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow', 'L_wrist',
                              'R_wrist', 'L_hip', 'R_hip', 'L_knee', 'R_knee', 'L_ankle', 'R_ankle', 'L_big_toe',
                              'R_big_toe', 'L_small_toe', 'R_small_toe', 'L_heel', 'R_heel', 'L_hand_thumb', 'R_hand_thumb',
                              'L_hand_index', 'R_hand_index', 'L_hand_middle', 'R_hand_middle', 'L_hand_pinky', 'R_hand_pinky']

                joints = {}
                for i, kp_3d in enumerate(body.keypoint):
                    organ = arr_organs[i]
                    joint = Joint(organ, kp_3d)
                    joints[organ] = joint

                    # Print the joint name and its coordinates
                    print(f"{organ}: X: {kp_3d[0]:.2f}, Y: {kp_3d[1]:.2f}, Z: {kp_3d[2]:.2f}")

                return joints

            else:
                time.sleep(0.01)
                return None

        else:
            return None

if __name__ == '__main__':
    s.camera_num = 1  # 0 - webcam, 2 - second USB in maya's computer

    # Audio variables initialization
    language = 'Hebrew'
    gender = 'Male'
    s.audio_path = 'audio files/' + language + '/' + gender + '/'
    s.picture_path = 'audio files/' + language + '/' + gender + '/'
    # s.str_to_say = ""
    # current_time = datetime.datetime.now()
    # s.participant_code = str(current_time.day) + "." + str(current_time.month) + " " + str(current_time.hour) + "." + \
    # str(current_time.minute) + "." + str(current_time.second)

    # Training variables initialization
    s.rep = 10
    s.waved = False
    s.success_exercise = False
    s.calibration = False
    s.finish_workout = False
    s.gymmy_done = False
    s.camera_done = False
    s.robot_count = False
    s.demo_finish = False
    s.list_effort_each_exercise = {}
    s.ex_in_training = []
    s.finish_program= False
    # s.exercises_start=False
    s.waved_has_tool = True  # True just in order to go through the loop in Gymmy
    s.max_repetitions_in_training=0
    # Excel variable
    ############################# להוריד את הסולמיות
    s.ex_list = {}
    #s.req_exercise = "bend_elbows_ball"
    time.sleep(2)
    # Create all components
    s.camera = Help()
    s.number_of_repetitions_in_training=0


    # Start all threads
    s.camera.start()

