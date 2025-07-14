import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import math
import threading
import socket
import json
import math
import time
import pandas as pd
import numpy as np
from statistics import mean, stdev
import datetime
import pyrealsense2 as rs

# internal imports
from MP import MP
from Joint import Joint
import Settings as s
import Excel
from Audio import say
#from performance_classification import feature_extraction, predict_performance, plot_data

class Camera(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.running = False
        self.req_exercise = None  # New: store exercise name for threading
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def start_camera(self,ex):
        self.req_exercise=ex
        self.start()

    def stop(self):
        self.pipeline.stop()
        self.pose.close()
        cv2.destroyAllWindows()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        image = np.asanyarray(color_frame.get_data())
        return image

    def get_skeleton_data(self):
        image = self.get_frame()
        if image is None:
            return None, None

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        if not results.pose_landmarks:
            return image, None

        h, w, _ = image.shape
        joints = {}

        for idx, lm in enumerate(results.pose_landmarks.landmark):
            x, y, z = int(lm.x * w), int(lm.y * h), int(lm.z * 100)
            joints[idx] = (x, y, z)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        # Draw connections
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx in joints and end_idx in joints:
                start_point = joints[start_idx][:2]
                end_point = joints[end_idx][:2]
                cv2.line(image, start_point, end_point, (255, 255, 255), 2)

        return image, joints

    def get_joint_name(self,joint_id):
        joint_names = {
            0: "Nose",
            1: "Left eye inner",
            2: "Left eye",
            3: "Left eye outer",
            4: "Right eye inner",
            5: "Right eye",
            6: "Right eye outer",
            7: "Left ear",
            8: "Right ear",
            9: "Mouth left",
            10: "Mouth right",
            11: "Left shoulder",
            12: "Right shoulder",
            13: "Left elbow",
            14: "Right elbow",
            15: "Left wrist",
            16: "Right wrist",
            17: "Left pinky",
            18: "Right pinky",
            19: "Left index",
            20: "Right index",
            21: "Left thumb",
            22: "Right thumb",
            23: "Left hip",
            24: "Right hip",
            25: "Left knee",
            26: "Right knee",
            27: "Left ankle",
            28: "Right ankle",
            29: "Left heel",
            30: "Right heel",
            31: "Left foot index",
            32: "Right foot index"
        }
        return joint_names.get(joint_id, "Unknown")

    def get_joint_id(self,joint_name):
        name_to_id = {
            "nose": 0,
            "left_eye_inner": 1,
            "left_eye": 2,
            "left_eye_outer": 3,
            "right_eye_inner": 4,
            "right_eye": 5,
            "right_eye_outer": 6,
            "left_ear": 7,
            "right_ear": 8,
            "mouth_left": 9,
            "mouth_right": 10,
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_pinky": 17,
            "right_pinky": 18,
            "left_index": 19,
            "right_index": 20,
            "left_thumb": 21,
            "right_thumb": 22,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
            "left_heel": 29,
            "right_heel": 30,
            "left_foot_index": 31,
            "right_foot_index": 32
        }
        return name_to_id.get(joint_name.lower())

    def calculate_angle(self, joints, joint1_id, joint2_id, joint3_id):
        if joint1_id not in joints or joint2_id not in joints or joint3_id not in joints:
            return None

        a = np.array(joints[joint1_id])
        b = np.array(joints[joint2_id])
        c = np.array(joints[joint3_id])

        ba = a - b
        bc = c - b

        ba_norm = ba / np.linalg.norm(ba)
        bc_norm = bc / np.linalg.norm(bc)

        cosine_angle = np.dot(ba_norm, bc_norm)
        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle_rad)

    def calc_angle_3d(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        ba_norm = ba / np.linalg.norm(ba)
        bc_norm = bc / np.linalg.norm(bc)

        cosine_angle = np.dot(ba_norm, bc_norm)
        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle_rad)

    def calc_dist(self, joint1, joint2):
        return math.sqrt(
            (joint1[0] - joint2[0]) ** 2 +
            (joint1[1] - joint2[1]) ** 2 +
            (joint1[2] - joint2[2]) ** 2
        )


    def exercise_two_angles_3d(self, exercise_name, joint1, joint2, joint3, up_lb, up_ub, down_lb, down_ub,
                               joint4, joint5, joint6, up_lb2, up_ub2, down_lb2, down_ub2, angle_classification, use_alternate_angles=False):
        flag = True
        counter = 0
        said_instructions = False
        list_joints = []
        while s.req_exercise == exercise_name:
            image, joints = self.get_skeleton_data()
            if joints is not None:
                # Convert joint names to IDs
                r_j1 = self.get_joint_id(str("right_" + joint1.lower()))
                r_j2 = self.get_joint_id(str("right_" + joint2.lower()))
                r_j3 = self.get_joint_id("right_" + joint3.lower())
                l_j1 = self.get_joint_id("left_" + joint1.lower())
                l_j2 = self.get_joint_id("left_" + joint2.lower())
                l_j3 = self.get_joint_id("left_" + joint3.lower())

                r_j4 = self.get_joint_id("right_" + joint4.lower())
                r_j5 = self.get_joint_id("right_" + joint5.lower())
                r_j6 = self.get_joint_id("right_" + joint6.lower())
                l_j4 = self.get_joint_id("left_" + joint4.lower())
                l_j5 = self.get_joint_id("left_" + joint5.lower())
                l_j6 = self.get_joint_id("left_" + joint6.lower())

                # Calculate angles
                right_angle = self.calc_angle_3d(joints[r_j1], joints[r_j2], joints[r_j3])
                left_angle = self.calc_angle_3d(joints[l_j1], joints[l_j2], joints[l_j3])

                if use_alternate_angles:
                    right_angle2 = self.calc_angle_3d(joints[l_j4], joints[r_j5], joints[r_j6])
                    left_angle2 = self.calc_angle_3d(joints[r_j4], joints[l_j5], joints[l_j6])
                else:
                    right_angle2 = self.calc_angle_3d(joints[r_j4], joints[r_j5], joints[r_j6])
                    left_angle2 = self.calc_angle_3d(joints[l_j4], joints[l_j5], joints[l_j6])

                new_entry = [joints[self.get_joint_id(str("right_" + joint1.lower()))], joints[self.get_joint_id(str("right_" + joint2.lower()))], joints[self.get_joint_id(str("right_" + joint3.lower()))],
                             joints[self.get_joint_id(str("left_" + joint1.lower()))], joints[self.get_joint_id(str("left_" + joint2.lower()))], joints[self.get_joint_id(str("left_" + joint3.lower()))],
                             joints[self.get_joint_id(str("right_" + joint4.lower()))], joints[self.get_joint_id(str("right_" + joint5.lower()))], joints[self.get_joint_id(str("right_" + joint6.lower()))],
                             joints[self.get_joint_id(str("left_" + joint4.lower()))], joints[self.get_joint_id(str("left_" + joint5.lower()))], joints[self.get_joint_id(str("left_" + joint6.lower()))],
                             right_angle, left_angle, right_angle2, left_angle2]
                list_joints.append(new_entry)
                if s.one_hand != False:
                    if s.one_hand == 'right':
                        if not flag:
                            left_angle = up_ub-1
                            left_angle2 = up_ub2-1
                        else:
                            left_angle = down_ub-1
                            left_angle2 = down_ub2-1
                    else: # s.one_hand = left
                        if not flag:
                            right_angle = up_ub-1
                            right_angle2 = up_ub2-1
                        else:
                            right_angle = down_ub-1
                            right_angle2 = down_ub2-1
                if right_angle is not None and left_angle is not None and \
                        right_angle2 is not None and left_angle2 is not None:
                    if (up_lb < right_angle < up_ub) & (up_lb < left_angle < up_ub) & \
                            (up_lb2 < right_angle2 < up_ub2) & (up_lb2 < left_angle2 < up_ub2) & (not flag):
                        flag = True
                        counter += 1
                        print(counter)
                        if not s.robot_count:
                            say(str(counter))
                    if (down_lb < right_angle < down_ub) & (down_lb < left_angle < down_ub) & \
                            (down_lb2 < right_angle2 < down_ub2) & (down_lb2 < left_angle2 < down_ub2) & (flag):
                        flag = False
            if (not s.robot_count) and (counter == s.rep):
                s.req_exercise = ""
                s.success_exercise = True
                break
            if s.corrective_feedback and (s.robot_rep >= s.rep/2) and counter <=2 and not said_instructions:
                say(exercise_name + "_" + str(flag))
                said_instructions = True
                if flag:
                    print("Corrective feedback true - Try to raise your hands more")
                if not flag:
                    print("Corrective feedback false - Try to close your hands more")
            cv2.imshow("Exercise View", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if s.adaptive:
            try:
                if angle_classification == "first":
                    self.classify_performance(list_joints, exercise_name, 12, 13, counter)
                else:
                    self.classify_performance(list_joints, exercise_name, 14, 15, counter)
            except Exception as e:
                print (f"can't do classification {e}")
        if s.one_hand=='right':
            exercise_name = exercise_name[:-9]
        elif s.one_hand=='left':
            exercise_name = exercise_name[:-9]
        s.ex_list.append([exercise_name, counter])
        name = (exercise_name+str(time.time()))[:25]
        Excel.wf_joints(name, list_joints)

    def exercise_one_angle_3d(self, exercise_name, joint1, joint2, joint3, up_lb, up_ub, down_lb, down_ub,
                              use_alternate_angles=False):
        flag = True
        counter = 0
        said_instructions = False
        list_joints = []
        while s.req_exercise == exercise_name:
            image, joints = self.get_skeleton_data()
            if joints is not None:
                # Convert joint names to IDs
                r_j1 = self.get_joint_id("right_"+joint1.lower())
                r_j2 = self.get_joint_id("right_" + joint2.lower())
                r_j3 = self.get_joint_id("right_" + joint3.lower())
                l_j1 = self.get_joint_id("left_" + joint1.lower())
                l_j2 = self.get_joint_id("left_" + joint2.lower())
                l_j3 = self.get_joint_id("left_" + joint3.lower())

                # Use alternate joint combination if specified
                if use_alternate_angles:
                    right_angle = self.calc_angle_3d(joints[l_j1], joints[r_j2], joints[r_j3])
                    left_angle = self.calc_angle_3d(joints[r_j1], joints[l_j2], joints[l_j3])
                else:
                    right_angle = self.calc_angle_3d(joints[r_j1], joints[r_j2], joints[r_j3])
                    left_angle = self.calc_angle_3d(joints[l_j1], joints[l_j2], joints[l_j3])

                # Save joint coordinates and angles
                new_entry = [
                    joints[r_j1], joints[r_j2], joints[r_j3],
                    joints[l_j1], joints[l_j2], joints[l_j3],
                    right_angle, left_angle
                ]
                #print("just saved your joints there")

                list_joints.append(new_entry)
                if right_angle is not None and left_angle is not None:
                    if (up_lb < right_angle < up_ub) & (up_lb < left_angle < up_ub) & (not flag):
                        flag = True
                        counter += 1
                        print(counter)
                        if not s.robot_count:
                            say(str(counter))
                    if (down_lb < right_angle < down_ub) & (down_lb < left_angle < down_ub) & (flag):
                        flag = False
            if (not s.robot_count) and (counter >= s.rep):
                print("Finish")
                s.req_exercise = ""
                s.success_exercise = True

                break
            if s.corrective_feedback and (s.robot_rep >= s.rep/2) and counter <=2 and not said_instructions:
                say(exercise_name + "_" + str(flag))
                said_instructions = True
                if flag:
                    print("Try to raise your hands more")
                if not flag:
                    print("Try to close your hands more")

            cv2.imshow("Exercise View", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if s.adaptive:
            self.classify_performance(list_joints, exercise_name, 6, 7, counter)
        s.ex_list.append([exercise_name, counter])
        name = (exercise_name+str(time.time()))[:25]
        Excel.wf_joints(name, list_joints)

# ------------------------------------ My Stuff -------------------------------------
    def notool_reverse_fly(self):
        self.exercise_two_angles_3d("notool_reverse_fly","Hip","Shoulder","Elbow",70,100,70,100,
                                    "Shoulder","Shoulder","Elbow",150,180,50,90,"first",True)

    def vertical_skullcrusher(self):
        self.exercise_two_angles_3d("vertical_skullcrusher","Hip","Shoulder","Elbow",130,170,130,170,
                                    "Shoulder","Elbow","Wrist",90,150,20,45,"first",True)


    def raise_arms_horizontally(self):
        self.exercise_two_angles_3d("raise_arms_horizontally", "Hip", "Shoulder", "Wrist", 80, 105, 5, 30,
                                    "Shoulder", "Shoulder", "Wrist", 150, 180, 80, 110, "first", True)

    def bend_elbows(self):
        self.exercise_one_angle_3d("bend_elbows", "Shoulder", "Elbow", "Wrist", 130, 180, 10, 60) #todo change to 2 angles - add armpit

    def raise_arms_bend_elbows(self):
        self.exercise_two_angles_3d("raise_arms_bend_elbows", "Shoulder", "Elbow", "Wrist", 130, 180, 10, 70,
                                    "Elbow", "Shoulder", "Hip", 60, 105, 60, 105, "first")

    def raise_arms_bend_elbows_one_hand(self):
        self.exercise_two_angles_3d("raise_arms_bend_elbows_one_hand", "Shoulder", "Elbow", "Wrist", 130, 180, 10, 70,
                                    "Elbow", "Shoulder", "Hip", 60, 105, 60, 105, "first")

    def open_and_close_arms(self):
        self.exercise_two_angles_3d("open_and_close_arms",  "Hip", "Shoulder", "Wrist", 80, 150, 80, 150,
                                   "Shoulder", "Shoulder", "Wrist", 90, 120, 150, 175, "second",  True)

    def open_and_close_arms_one_hand(self):
        self.exercise_two_angles_3d("open_and_close_arms_one_hand",  "Hip", "Shoulder", "Wrist", 80, 150, 80, 150,
                                   "Shoulder", "Shoulder", "Wrist", 90, 120, 150, 175, "second",  True)

    def open_and_close_arms_90(self):
        self.exercise_two_angles_3d("open_and_close_arms_90", "Wrist", "Elbow", "Shoulder", 60, 150, 60, 150,
                                    "Shoulder", "Shoulder", "Elbow", 140, 180, 80, 120, "second", True)

    def open_and_close_arms_90_one_hand(self):
        self.exercise_two_angles_3d("open_and_close_arms_90_one_hand", "Wrist", "Elbow", "Shoulder", 60, 150, 60, 150,
                                    "Shoulder", "Shoulder", "Elbow", 140, 180, 80, 120, "second", True)

    def raise_arms_forward(self):
        self.exercise_two_angles_3d("raise_arms_forward", "Wrist", "Shoulder", "Hip", 85, 135, 10, 50,
                                   "Shoulder", "Shoulder", "Wrist", 80, 115, 80, 115, "first", True)

    def raise_arms_forward_one_hand(self):
        self.exercise_two_angles_3d("raise_arms_forward_one_hand", "Wrist", "Shoulder", "Hip", 85, 135, 10, 50,
                                   "Shoulder", "Shoulder", "Wrist", 80, 115, 80, 115, "first", True)
    def hello_waving(self):
        self.exercise_one_angle_3d("hello_waving","Shoulder","Elbow","Wrist",20,60,70,180)
        s.waved=True

    def init_position(self):
        if not self.running:  # Ensure camera is started
            self.pipeline.start(self.config)
            self.running = True

        init_pos = False
        say("calibration")
        print("CAMERA: init position - please stand in front of the camera with hands to the sides")

        while not init_pos:
            image, joints = self.get_skeleton_data()
            if joints is not None:
                count = 0
                for j in joints.values():
                    if j is not None:
                        count += 1

                r_shoulder = self.get_joint_id("right_shoulder")
                r_hip = self.get_joint_id("right_hip")
                r_wrist = self.get_joint_id("right_wrist")
                l_shoulder = self.get_joint_id("left_shoulder")
                l_hip = self.get_joint_id("left_hip")
                l_wrist = self.get_joint_id("left_wrist")

                required = [r_shoulder, r_hip, r_wrist, l_shoulder, l_hip, l_wrist]

                if all(j in joints for j in required):
                    angle_right = self.calc_angle_3d(joints[r_shoulder], joints[r_hip], joints[r_wrist])
                    angle_left = self.calc_angle_3d(joints[l_shoulder], joints[l_hip], joints[l_wrist])

                    if count == len(joints) and angle_right > 80 and angle_left > 80:
                        init_pos = True
                        print("CAMERA: init position verified")
                else:
                    print("CAMERA: some key joints not detected.")
            else:
                print("CAMERA: user not detected")

            cv2.imshow("Calibration View", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        s.calibration = True
        say("calibration_complete")
        cv2.destroyWindow("Calibration View")

    #    def hello_waving(self):  # check if the participant waved
#        time.sleep(4)
#        print("Camera: Wave for start")
#        say('ready wave')
#        while s.req_exercise == "hello_waving":
#            joints = self.get_skeleton_data()
#            if joints is not None:
#                right_shoulder = joints[self.get_joint_id(str("right_shoulder"))]
#                right_wrist = joints[self.get_joint_id(str("right_wrist"))]
#                if right_shoulder.y < right_wrist.y != 0:
                    # print(right_shoulder.y)
                    # print(right_wrist.y)
#                    s.waved = True
#                    s.req_exercise = ""

    def classify_performance(self, list_joints, exercise_name, index_angle_right, index_angle_left, counter):
        if counter > 1:
            try:
                df = pd.DataFrame([entry[index_angle_right:index_angle_left + 1] for entry in list_joints]).T

                right_hand_data = df.iloc[0].dropna().to_numpy()
                left_hand_data = df.iloc[1].dropna().to_numpy()

                #features = feature_extraction(right_hand_data, left_hand_data)
                #predictions = predict_performance(features, exercise_name, s.adaptation_model_name)

                timestamp_key = exercise_name
                if exercise_name in s.performance_class:
                    current_time = datetime.datetime.now()
                    timestamp_key += str(current_time.minute) + str(current_time.second)

                s.performance_class[timestamp_key] = {
#                    'right': predictions[1],
#                    'left': predictions[0]
                }

                print(f"[Camera] Performance classified: {s.performance_class}")
#                plot_data(exercise_name, right_hand_data, left_hand_data)

            except Exception as e:
                print(f"[Camera] Error during classification: {e}")

        else:
            s.performance_class[exercise_name] = {'right': 1, 'left': 1}  # Default if not enough rep

    def run(self):
       self.running=True
       try:
           print("CAMERA STARTED")
           while not s.finish_workout:
               #time.sleep(0.00000001)
               image, joints = self.get_skeleton_data()
               if image is None:
                   continue
               if joints:
                   if s.req_exercise == "hello_waving":
                       self.hello_waving()
                       s.req_exercise = ""
                       s.camera_done = True

                   else:
                    if s.req_exercise != "":
                        print("CAMERA: Exercise", s.req_exercise," ", s.rep, "reps; start")
                        time.sleep(1)
                        getattr(self, s.req_exercise)()
                        print("CAMERA: Exercise ", s.req_exercise, " done")
                        s.req_exercise = ""
                        s.camera_done = True

           print("Camera Done")

       finally:
           self.stop()


if __name__ == "__main__":
    language = 'Hebrew'
    gender = 'Female'
    s.audio_path = 'audio files/' + language + '/' + gender + '/'
    s.finish_workout = False
    s.participant_code = "1106"
    s.rep = 2
    s.corrective_feedback = False
    s.one_hand = 'right'
    s.req_exercise = "hello_waving"
    s.robot_count = False
    Excel.create_workbook()
    s.ex_list = []
    s.adaptive = True
    if s.adaptive:
        s.adaptation_model_name = 'model2_resaved'
        s.performance_class = {}
    print('HelloServer')
    camera = Camera()
    camera.run()
