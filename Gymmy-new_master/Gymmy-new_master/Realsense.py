import json
import random
import pyzed.sl as sl
import threading
import socket
from Audio import say, get_wav_duration
from Joint import Joint
from MP import MP
from PyZedWrapper import PyZedWrapper
import Settings as s
import time
import Excel
from ScreenNew import Screen, FullScreenApp
import numpy as np
from openpyxl import Workbook
from scipy.signal import savgol_filter

class KalmanFilter:
    def __init__(self, initial_position, process_noise=1e-2, measurement_noise=1e-1, error_estimate=1, alpha=0.5, threshold=20):
        self.current_estimate = np.array(initial_position)  # Current estimate
        self.estimate_covariance = error_estimate  # Error estimate
        self.process_noise = process_noise  # Process noise
        self.measurement_noise = measurement_noise  # Measurement noise
        self.alpha = alpha  # EMA smoothing factor
        self.smoothed_estimate = np.array(initial_position)  # Smoothed estimate
        self.previous_positions = [initial_position]  # Keep track of previous positions
        self.threshold = threshold  # Threshold for large intervals

    def predict(self):
        self.current_estimate = self.current_estimate
        self.estimate_covariance += self.process_noise

    def update(self, measurement):
        if len(self.previous_positions) > 0:
            prev_position = self.previous_positions[-1]
            distance = np.linalg.norm(np.array(measurement) - np.array(prev_position))

            # Debugging the distances
            # print(f"Distance between frames: {distance}")
            # print(f"Measurement: {measurement}, Previous Position: {prev_position}")

            if distance > self.threshold:
               # print("Large interval detected. Smoothing...")

                # Only smooth if the distance truly exceeds the threshold
                measurement = self.average_previous_positions()

        # Apply Savitzky-Golay filter for smoothing
        measurement = self.apply_savgol_filter(measurement)

        kalman_gain = self.estimate_covariance / (self.estimate_covariance + self.measurement_noise)
        self.current_estimate += kalman_gain * (np.array(measurement) - self.current_estimate)
        self.estimate_covariance *= (1 - kalman_gain)
        self.smoothed_estimate = self.apply_ema(self.smoothed_estimate, self.current_estimate)

        # Add the current position to the list of previous positions
        self.previous_positions.append(measurement)
        if len(self.previous_positions) > 5:  # Limit the number of stored previous positions
            self.previous_positions.pop(0)

    def apply_ema(self, previous_estimate, current_estimate):
        return self.alpha * current_estimate + (1 - self.alpha) * previous_estimate

    def average_previous_positions(self):
        # Average the previous few positions to smooth the transition
        return np.mean(self.previous_positions[-3:], axis=0)

    def get_estimate(self):
        return self.smoothed_estimate

    def apply_savgol_filter(self, measurement, window_length=15, polyorder=2):
        # Ensure there are enough points to apply the filter
        if len(self.previous_positions) >= window_length:
            # Apply the Savitzky-Golay filter to the last few positions (including current)
            smoothed = savgol_filter(np.array(self.previous_positions[-window_length:] + [measurement]), window_length, polyorder, axis=0)
            return smoothed[-1]  # Return the last smoothed value (current position)
        else:
            return measurement  # If not enough data, return the original measurement


class Realsense(threading.Thread):

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

    def __init__(self):
        threading.Thread.__init__(self)
        # Initialize the ZED camera or RealSense (depends on your setup)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = ('localhost', 7000)
        self.sock.bind(self.server_address)
        print("CAMERA INITIALIZATION")
        self.frame_count = 0
        self.start_time = None
        self.kalman_filters = {}


    def run(self):
        print("CAMERA START")
        medaip = MP()  # Assuming PyZedWrapper manages your camera setup
        medaip.start()
        while not s.finish_program:
            time.sleep(0.0001)  # Prevents the MP from being stuck
            if s.req_exercise != "":
                ex = s.req_exercise
                print("CAMERA: Exercise", ex, "start")
                if s.req_exercise != "hello_waving":
                    audio = s.req_exercise
                    time.sleep(get_wav_duration(audio) + get_wav_duration("start_ex"))
                    s.max_repetitions_in_training += s.rep  # Number of repetitions expected in this exercise
                getattr(self, ex)()  # Call the exercise method dynamically
                print("CAMERA: Exercise", ex, "done")
                s.req_exercise = ""
                s.camera_done = True
            else:
                time.sleep(1)  # Prevents the MP from being stuck
        print("Camera Done")

    # def get_skeleton_data(self):
    #     self.sock.settimeout(1)
    #     try:
    #         data, address = self.sock.recvfrom(4096)
    #         # Decode and split the received data
    #         data = json.loads(data.decode())
    #         data = data.split('/')
    #
    #         joints_str = []
    #         for i in data:
    #             joint_data = i.split(',')
    #             joints_str.append(joint_data)
    #
    #         joints_str = joints_str[:-1]  # Remove the empty entry at the end
    #
    #         joints = {}  # Dictionary to store joint data
    #
    #         for j in joints_str:
    #             joint_name = j[0]
    #             joint_position = [float(j[1]), float(j[2]), float(j[3]) * 100]  # z is scaled by 100
    #
    #             print(f"Processing joint: {joint_name}, Position: {joint_position}")
    #
    #             Store [0, 0, 0] as the joint position but skip processing it
    #             if joint_position == [0.0, 0.0, 0.0]:
    #                 #print(f"Skipping Kalman filter update for {joint_name} due to invalid position.")
    #                 # Directly store the invalid position
    #                 joints[joint_name] = Joint(joint_name, 0.0, 0.0, 0.0)
    #                 continue
    #
    #             # Initialize Kalman filter for this joint if it doesn't already exist
    #             if joint_name not in self.kalman_filters:
    #                 self.kalman_filters[joint_name] = KalmanFilter(initial_position=joint_position)
    #                 #print(f"Initializing Kalman filter for {joint_name} with position {joint_position}")
    #
    #             # Get the Kalman filter for this joint
    #             kalman_filter = self.kalman_filters[joint_name]
    #
    #             # Apply Kalman filter: predict and update with valid data
    #             kalman_filter.predict()  # Kalman filter prediction step
    #             kalman_filter.update(joint_position)  # Update with the new measurement
    #             filtered_position = kalman_filter.get_estimate()  # Get the filtered position
    #
    #             Store the filtered joint data
    #             joints[joint_name] = Joint(joint_name, filtered_position[0], filtered_position[1], filtered_position[2])
    #
    #         return joints
    #
    #     except socket.timeout:
    #         print("Didn't receive data! [Timeout]")
    #         return None

            #
            # for key, value in lm_dict.items():
            #     joint_data = [float(joints_str[value][1]), float(joints_str[value][2]), float(joints_str[value][3])*100]
            #
            #     if key in joints:
            #         # If joint already exists, update using Kalman Filter
            #         joints[key].kalman_filter.predict()  # Predict the next position
            #         joints[key].kalman_filter.update(joint_data)  # Update the Kalman filter with the new data
            #         joints[key].position = joints[key].kalman_filter.get_estimate()  # Get the smoothed position
            #     else:
            #         # Create a new joint if it doesn't exist
            #         joint = Joint(key, joint_data)
            #         joint.kalman_filter = KalmanFilter(joint_data, alpha=0.5)  # Initialize Kalman Filter
            #         joints[key] = joint

    def get_skeleton_data(self):
        self.sock.settimeout(1)
        try:
            data, address = self.sock.recvfrom(4096)
            # print('received {} bytes from {}'.format(len(data), address))
            data = json.loads(data.decode())
            data = data.split('/')
            joints_str = []
            for i in data:
                joint_data = i.split(',')
                joints_str.append(joint_data)
            joints_str = joints_str[:-1]  # remove the empty list at the end
            # change from string to float values
            joints = {}  # joints dict data
            for j in joints_str:
                joints[j[0]] = Joint(j[0], float(j[1]), float(j[2]), float(j[3]) * 100)  # z is smaller than x and y!!
            return joints
        except socket.timeout:  # fail after 1 second of no activity
            print("Didn't receive data! [Timeout]")
            return None






    def exercise_two_angles_3d(self, exercise_name, joint1, joint2, joint3, up_lb, up_ub, down_lb, down_ub,
                                   joint4, joint5, joint6, up_lb2, up_ub2, down_lb2, down_ub2, use_alternate_angles=False, left_right_differ=False):

            list_first_angle=[]
            list_second_angle=[]
            flag = True
            counter = 0
            list_joints = []
            while s.req_exercise == exercise_name:
                while s.did_training_paused:
                    time.sleep(0.01)
            #for i in range (1,200):
                joints = self.get_skeleton_data()
                if joints is not None:
                    right_angle = self.calc_angle_3d(joints[str("R_" + joint1)], joints[str("R_" + joint2)],
                                                     joints[str("R_" + joint3)])
                    left_angle = self.calc_angle_3d(joints[str("L_" + joint1)], joints[str("L_" + joint2)],
                                                    joints[str("L_" + joint3)])
                    if use_alternate_angles:
                        right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)],
                                                         joints[str("L_" + joint6)])
                        left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)],
                                                         joints[str("R_" + joint6)])

                        new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                     joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                     joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("L_" + joint6)],
                                     joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)],
                                     right_angle, left_angle, right_angle2, left_angle2]

                    else:
                        right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)],
                                                       joints[str("R_" + joint6)])
                        left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)],
                                                      joints[str("L_" + joint6)])

                        new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                     joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                     joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("R_" + joint6)],
                                     joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("L_" + joint6)],
                                     right_angle, left_angle, right_angle2, left_angle2]

                    ##############################################################################
                    print(left_angle, " ", right_angle)
                    print(left_angle2, " ", right_angle2)

                    list_first_angle.append(left_angle)
                    list_second_angle.append(left_angle2)

                    ##############################################################################



                    list_joints.append(new_entry)
                    #print(str(i))
                    if right_angle is not None and left_angle is not None and \
                            right_angle2 is not None and left_angle2 is not None:
                        print("first angle mean: ", np.nanmean(list_first_angle))
                        print("first angle stdev: ", np.nanstd(list_first_angle))
                        print("second angle mean: ", np.nanmean(list_second_angle))
                        print("second angle stdev: ", np.nanstd(list_second_angle))


                        if left_right_differ:
                            if (up_lb < right_angle < up_ub) & (down_lb < left_angle < down_ub) & \
                                    (up_lb2 < right_angle2 < up_ub2) & (down_lb2 < left_angle2 < down_ub2) & (not flag):
                                flag = True
                                counter += 1
                                s.number_of_repetitions_in_training += 1
                                s.patient_repetitions_counting_in_exercise+=1
                                #self.change_count_screen(counter)
                                print("counter:"+ str(counter))
                              #  if not s.robot_count:
                                say(str(counter))
                            elif (down_lb < right_angle < down_ub) & (up_lb < left_angle < up_ub) & \
                                    (down_lb2 < right_angle2 < down_ub2) & (up_lb2 < left_angle2 < up_ub2) & (flag):
                                flag = False


                        else:
                            if (up_lb < right_angle < up_ub) & (up_lb < left_angle < up_ub) & \
                                    (up_lb2 < right_angle2 < up_ub2) & (up_lb2 < left_angle2 < up_ub2) & (not flag):
                                flag = True
                                counter += 1
                                s.number_of_repetitions_in_training += 1
                                s.patient_repetitions_counting_in_exercise+=1
                                #self.change_count_screen(counter)
                                print("counter:" + str(counter))
                                #  if not s.robot_count:
                                say(str(counter))
                            elif (down_lb < right_angle < down_ub) & (down_lb < left_angle < down_ub) & \
                                    (down_lb2 < right_angle2 < down_ub2) & (down_lb2 < left_angle2 < down_ub2) & (flag):
                                flag = False

                if counter == s.rep:
                    s.req_exercise = ""
                    s.success_exercise = True
                    break
            #self.ezer(list_first_angle)
            #self.end_exercise(counter)
            s.ex_list.update({exercise_name: counter})
            Excel.wf_joints(exercise_name, list_joints)


    def exercise_two_angles_3d_with_axis_check(self, exercise_name, joint1, joint2, joint3, up_lb, up_ub, down_lb, down_ub,
                               joint4, joint5, joint6, up_lb2, up_ub2, down_lb2, down_ub2, diff_size, use_alternate_angles=False,
                               left_right_differ=False):

        list_first_angle = []
        list_second_angle = []
        flag = True
        counter = 0
        list_joints = []
        while s.req_exercise == exercise_name:
            while s.did_training_paused:
                time.sleep(0.01)
        #for i in range (1,100):
            joints = self.get_skeleton_data()
            if joints is not None:
                right_angle = self.calc_angle_3d(joints[str("R_" + joint1)], joints[str("R_" + joint2)],
                                                 joints[str("R_" + joint3)])
                left_angle = self.calc_angle_3d(joints[str("L_" + joint1)], joints[str("L_" + joint2)],
                                                joints[str("L_" + joint3)])
                if use_alternate_angles:
                    right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)],
                                                      joints[str("L_" + joint6)])
                    left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)],
                                                     joints[str("R_" + joint6)])

                    new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                 joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                 joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("L_" + joint6)],
                                 joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)],
                                 right_angle, left_angle, right_angle2, left_angle2]


                else:
                    right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)],
                                                      joints[str("R_" + joint6)])
                    left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)],
                                                     joints[str("L_" + joint6)])

                    new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                 joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                 joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("R_" + joint6)],
                                 joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("L_" + joint6)],
                                 right_angle, left_angle, right_angle2, left_angle2]

                ##############################################################################
                print(left_angle, " ", right_angle)
                print(left_angle2, " ", right_angle2)

                list_first_angle += [left_angle]
                list_second_angle += [left_angle2]

                print("left shoulder", joints["L_shoulder"].__str__())
                print("right shoulder", joints["R_shoulder"].__str__())
                print(str(abs(joints["L_shoulder"].x - joints["R_shoulder"].x)))


                ##############################################################################


                #print(i)
                list_joints.append(new_entry)
                if right_angle is not None and left_angle is not None and \
                        right_angle2 is not None and left_angle2 is not None:
                    print("first angle mean: ", np.nanmean(list_first_angle))
                    print("first angle stdev: ", np.nanstd(list_first_angle))
                    print("second angle mean: ", np.nanmean(list_second_angle))
                    print("second angle stdev: ", np.nanstd(list_second_angle))

                    if left_right_differ:
                        if (up_lb < right_angle < up_ub) & (down_lb < left_angle < down_ub) & \
                                (up_lb2 < right_angle2 < up_ub2) & (down_lb2 < left_angle2 < down_ub2) & \
                                (abs(joints["L_shoulder"].x - joints["R_shoulder"].x) < 200) & (not flag):
                            flag = True
                            counter += 1
                            s.number_of_repetitions_in_training += 1
                            s.patient_repetitions_counting_in_exercise += 1
                            #self.change_count_screen(counter)
                            print("counter:" + str(counter))
                            #  if not s.robot_count:
                            say(str(counter))
                        elif (down_lb < right_angle < down_ub) & (up_lb < left_angle < up_ub) & \
                                (down_lb2 < right_angle2 < down_ub2) & (up_lb2 < left_angle2 < up_ub2) & \
                                (abs(joints["L_shoulder"].x - joints["R_shoulder"].x) < 200) & (flag):
                            flag = False


                    else:
                        if (up_lb < right_angle < up_ub) & (up_lb < left_angle < up_ub) & \
                                (up_lb2 < right_angle2 < up_ub2) & (up_lb2 < left_angle2 < up_ub2) & (not flag):
                            flag = True
                            counter += 1
                            s.number_of_repetitions_in_training += 1
                            s.patient_repetitions_counting_in_exercise += 1
                            #self.change_count_screen(counter)
                            print("counter:" + str(counter))
                            #  if not s.robot_count:
                            say(str(counter))
                        elif (down_lb < right_angle < down_ub) & (down_lb < left_angle < down_ub) & \
                                (down_lb2 < right_angle2 < down_ub2) & (down_lb2 < left_angle2 < down_ub2) & (flag):
                            flag = False

            if counter == s.rep:
                s.req_exercise = ""
                s.success_exercise = True
                break

        #self.end_exercise(counter)
        s.ex_list.update({exercise_name: counter})
        Excel.wf_joints(exercise_name, list_joints)

    def exercise_three_angles_3d(self, exercise_name, joint1, joint2, joint3, up_lb, up_ub, down_lb, down_ub,
                               joint4, joint5, joint6, up_lb2, up_ub2, down_lb2, down_ub2,
                                joint7, joint8, joint9, up_lb3, up_ub3, down_lb3, down_ub3, use_alternate_angles=False,
                               left_right_differ=False):


        flag = True
        counter = 0
        list_joints = []
        while s.req_exercise == exercise_name:
            while s.did_training_paused:
                time.sleep(0.01)
        #for i in range (1,100):
            joints = self.get_skeleton_data()
            if joints is not None:
                right_angle = self.calc_angle_3d(joints[str("R_" + joint1)], joints[str("R_" + joint2)],
                                                 joints[str("R_" + joint3)])
                left_angle = self.calc_angle_3d(joints[str("L_" + joint1)], joints[str("L_" + joint2)],
                                                joints[str("L_" + joint3)])

                right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)],
                                                 joints[str("R_" + joint6)])
                left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)],
                                                joints[str("L_" + joint6)])

                if use_alternate_angles:
                    right_angle3 = self.calc_angle_3d(joints[str("R_" + joint7)], joints[str("R_" + joint8)],
                                                      joints[str("L_" + joint9)])
                    left_angle3 = self.calc_angle_3d(joints[str("L_" + joint7)], joints[str("L_" + joint8)],
                                                     joints[str("R_" + joint9)])

                    new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                 joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                 joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("R_" + joint6)],
                                 joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("L_" + joint6)],
                                 joints[str("R_" + joint7)], joints[str("R_" + joint8)], joints[str("L_" + joint9)],
                                 joints[str("L_" + joint7)], joints[str("L_" + joint8)], joints[str("R_" + joint9)],
                                 right_angle, left_angle, right_angle2, left_angle2, right_angle3, left_angle3]

                else:
                    right_angle3 = self.calc_angle_3d(joints[str("R_" + joint7)], joints[str("R_" + joint8)],
                                                      joints[str("R_" + joint9)])
                    left_angle3 = self.calc_angle_3d(joints[str("L_" + joint7)], joints[str("L_" + joint8)],
                                                     joints[str("L_" + joint9)])

                    new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                 joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                 joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("R_" + joint6)],
                                 joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("L_" + joint6)],
                                 joints[str("R_" + joint7)], joints[str("R_" + joint8)], joints[str("R_" + joint9)],
                                 joints[str("L_" + joint7)], joints[str("L_" + joint8)], joints[str("L_" + joint9)],
                                 right_angle, left_angle, right_angle2, left_angle2, right_angle3, left_angle3]

                ##############################################################################
                print(left_angle, " ", right_angle)
                print(left_angle2, " ", right_angle2)
                print(left_angle3, " ", right_angle3)
                print("#######################################")
                ##############################################################################


                list_joints.append(new_entry)
                if right_angle is not None and left_angle is not None and \
                        right_angle2 is not None and left_angle2 is not None and \
                        right_angle3 is not None and left_angle3 is not None:

                    if (up_lb < right_angle < up_ub) & (up_lb < left_angle < up_ub) & \
                            (up_lb2 < right_angle2 < up_ub2) & (up_lb2 < left_angle2 < up_ub2) & \
                            (up_lb3 < right_angle3 < up_ub3) & (up_lb3 < left_angle3 < up_ub3) & (not flag):
                        flag = True
                        counter += 1
                        s.number_of_repetitions_in_training += 1
                        s.patient_repetitions_counting_in_exercise += 1
                        print("counter:" + str(counter))
                        #self.change_count_screen(counter)
                       # if not s.robot_count:
                        say(str(counter))
                    elif (down_lb < right_angle < down_ub) & (down_lb < left_angle < down_ub) & \
                            (down_lb2 < right_angle2 < down_ub2) & (down_lb2 < left_angle2 < down_ub2) & \
                            (down_lb3 < right_angle3 < down_ub3) & (down_lb3 < left_angle3 < down_ub3) & (flag):
                        flag = False

            if counter == s.rep:
                s.req_exercise = ""
                s.success_exercise = True
                break

        #self.end_exercise(counter)
        s.ex_list.update({exercise_name: counter})
        Excel.wf_joints(exercise_name, list_joints)


    def exercise_one_angle_3d_by_sides(self, exercise_name, joint1, joint2, joint3, one_lb, one_ub, two_lb, two_ub, side):
        flag = True
        counter = 0
        list_joints = []
        while s.req_exercise == exercise_name:
            while s.did_training_paused:
                time.sleep(0.01)
            joints = self.get_skeleton_data()
            if joints is not None:
                right_angle= self.calc_angle_3d(joints[str("R_" + joint1)], joints[str("R_" + joint2)],
                                          joints[str("R_" + joint3)])

                left_angle = self.calc_angle_3d(joints[str("L_" + joint1)], joints[str("L_" + joint2)],
                                         joints[str("L_" + joint3)])

                new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                             joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                             right_angle, left_angle]
                list_joints.append(new_entry)

                ##############################################################################
                print(left_angle, " ", right_angle)
                print("left wrist x: ", joints[str("R_wrist")].x)
                print("right wrist x: ", joints[str("L_shoulder")].x)
                print("nose: ", joints[str("nose")].y)


                ##############################################################################

                if side == 'right':
                    if right_angle is not None and left_angle is not None:
                        if (one_lb < right_angle < one_ub) & (joints[str("R_wrist")].x>joints[str("L_shoulder")].x+50) & (joints[str("nose")].y-50>joints[str("R_wrist")].y) & (not flag):
                            flag = True
                            counter += 1
                            s.patient_repetitions_counting_in_exercise += 1
                            s.number_of_repetitions_in_training += 1
                            print("counter:" + str(counter))
                            #self.change_count_screen(counter)
                            # if not s.robot_count:
                            say(str(counter))
                        elif (two_lb < right_angle < two_ub) & (joints[str("R_wrist")].x<joints[str("L_shoulder")].x-400) & (flag):
                            flag = False

                else:
                    if right_angle is not None and left_angle is not None:
                        if (one_lb < left_angle < one_ub) & (joints[str("R_shoulder")].x-50>joints[str("L_wrist")].x)& (joints[str("nose")].y - 50 > joints[str("L_wrist")].y) & (not flag):
                            flag = True
                            counter += 1
                            s.number_of_repetitions_in_training += 1
                            s.patient_repetitions_counting_in_exercise += 1
                            print("counter:" + str(counter))
                            #self.change_count_screen(counter)
                            #if not s.robot_count:
                            say(str(counter))
                        elif (two_lb < left_angle < two_ub) & (joints[str("L_wrist")].x>joints[str("R_shoulder")].x+400) & (flag):
                            flag = False

            if counter == s.rep:
                s.req_exercise = ""
                s.success_exercise = True
                break

        #self.end_exercise(counter)
        s.ex_list.update({exercise_name: counter})
        Excel.wf_joints(exercise_name, list_joints)

    def hello_waving(self):  # check if the participant waved
        # s.waved_has_tool = True
        # s.req_exercise = ""
        while s.req_exercise == "hello_waving":
            joints = self.get_skeleton_data()
            if joints is not None:
                right_shoulder = joints[str("R_shoulder")]
                right_wrist = joints[str("R_wrist")]
                print(f'shoulder: {str(right_shoulder.y)}')
                print(f'wrist: {str(right_wrist.y)}')
                if right_shoulder.y < right_wrist.y != 0:
                    print(f'xxxxxxxxxxxxxxxxxxxxxxxxxxxx: {str(right_shoulder.y)}')
                    print(f'xxxxxxxxxxxxxxxxxxxxxxxxxxxx: {str(right_wrist.y)}')
                    s.waved_has_tool = True
                    # print(right_shoulder.y)
                    # print(right_wrist.y)
                    s.req_exercise = ""


######################################################### First set of ball exercises

    def bend_elbows_ball(self):  # EX1
        self.exercise_two_angles_3d("bend_elbows_ball", "shoulder", "elbow", "wrist",150, 180, 10, 60,
                                    "elbow", "shoulder", "hip", 0, 60, 0, 60)

    def raise_arms_above_head_ball(self):  # EX2
        self.exercise_two_angles_3d("raise_arms_above_head_ball", "hip", "shoulder", "elbow", 125, 170, 0, 50,
                                    "shoulder", "elbow", "wrist", 120, 180, 135, 180)

    # def raise_arms_forward_turn_ball(self):  # EX3
    #     self.exercise_two_angles_3d_with_axis_check("raise_arms_forward_turn_ball", "shoulder", "elbow","wrist", 100, 180, 140, 180,
    #                                 "wrist", "hip", "hip",95,140,35,70, True, True)
    #                                 #"wrist", "hip", "hip",95 ,135 , 35, 70, True, True)


    def raise_arms_forward_turn_ball(self):  # EX3
        self.exercise_two_angles_3d("raise_arms_forward_turn_ball", "shoulder", "elbow","wrist", 100, 180, 140, 180,
                                    "wrist", "hip", "hip",95,140,35,70, True, True)
                                    #"wrist", "hip", "hip",95 ,135 , 35, 70, True, True)

######################################################### Second set of ball exercises

    def open_arms_and_forward_ball(self):  # EX4
        self.exercise_three_angles_3d("open_arms_and_forward_ball", "hip", "shoulder", "elbow", 40, 120, 80, 120,
                                      "shoulder", "elbow", "wrist",0,180,140,180,
                                    "elbow", "shoulder", "shoulder", 60, 135, 150,180,True)


    def open_arms_above_head_ball(self):  # EX5
        self.exercise_two_angles_3d("open_arms_above_head_ball", "elbow", "shoulder", "hip", 145,180, 80, 110,
                                   "shoulder", "elbow", "wrist", 130, 180, 130, 180)


########################################################### Set with a rubber band

    def open_arms_with_band(self):  # EX6
        self.exercise_two_angles_3d("open_arms_with_band","hip", "shoulder", "wrist", 85, 120, 70, 120,
                                    "wrist", "shoulder", "shoulder", 135,170,70,110,True)

        #"wrist", "shoulder", "shoulder", 100, 160,75, 95, True)

    def open_arms_and_up_with_band(self):  # EX7
        self.exercise_three_angles_3d("open_arms_and_up_with_band", "hip", "shoulder", "wrist", 125, 170, 20, 100,
                                    "shoulder", "elbow", "wrist", 130,180,0,180,
                                    "elbow", "shoulder", "shoulder", 110, 160, 70, 105, True)


    def up_with_band_and_lean(self):  # EX8
        self.exercise_two_angles_3d("up_with_band_and_lean", "shoulder", "elbow", "wrist", 125, 180, 125,180,
                                   "wrist", "hip", "hip", 120, 170, 50, 100, True, True)




######################################################  Set with a stick
    def bend_elbows_stick(self):  # EX9
        self.exercise_two_angles_3d("bend_elbows_stick", "shoulder", "elbow", "wrist",135, 180, 10, 40,
                                    "elbow", "shoulder", "hip", 0, 35, 0, 30)

    def bend_elbows_and_up_stick(self):  # EX10
        self.exercise_two_angles_3d("bend_elbows_and_up_stick", "hip", "shoulder", "elbow", 110, 170, 10, 50,
                                 "shoulder", "elbow", "wrist", 125, 180, 30, 85)

    def arms_up_and_down_stick(self):  # EX11
        self.exercise_two_angles_3d("arms_up_and_down_stick", "hip", "shoulder", "elbow", 115, 180, 10, 55,
                                    "wrist", "elbow", "shoulder", 130,180,130,180)

    def switch_with_stick(self):  # EX12
        self.exercise_two_angles_3d_with_axis_check("switch_with_stick", "shoulder", "elbow", "wrist", 0, 180, 140, 180,
                                    "wrist", "hip", "hip", 95, 140, 35, 70, True, True)

################################################# Set of exercises without equipment
    def hands_behind_and_lean_notool(self): # EX13
        self.exercise_two_angles_3d("hands_behind_and_lean_notool", "shoulder", "elbow", "wrist", 10,70,10,70,
                                    "elbow", "shoulder", "hip", 30, 95, 125, 170,False, True)

    #def hands_behind_and_turn_both_sides(self):  # EX14
     #   self.exercise_two_angles_3d("hands_behind_and_turn_both_sides", "elbow", "shoulder", "hip", 140,180,15,100,
      #                              "elbow", "hip", "knee", 130, 115, 80, 105, False, True)

    def right_hand_up_and_bend_notool(self):  # EX14
        self.exercise_one_angle_3d_by_sides("right_hand_up_and_bend_notool", "hip", "shoulder", "wrist", 120, 160, 0, 180, "right")

    def left_hand_up_and_bend_notool(self): #EX15
        self.exercise_one_angle_3d_by_sides("left_hand_up_and_bend_notool", "hip", "shoulder", "wrist", 120, 160, 0, 180, "left")

    def raising_hands_diagonally_notool(self): # EX16
        self.exercise_two_angles_3d_with_axis_check("raising_hands_diagonally_notool", "wrist", "shoulder", "hip", 0, 100, 105, 135,
                                    "elbow", "shoulder", "shoulder", 0, 180, 40, 75, True, True)\


    def help_function(self):
        while True:
            joints = self.get_skeleton_data()
            print(f'R_hand_thumb: {joints["R_hand_thumb"]} ')
            print(f'R_hand_pinky: {joints["R_hand_pinky"]} ')

            # arr_organs = ['pelvis', 'spine_1', 'spine_2', 'spine_3', 'neck', 'nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
            #               'L_clavicle', 'R_clavicle', 'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow', 'L_wrist',
            #               'R_wrist', 'L_hip', 'R_hip', 'L_knee', 'R_knee', 'L_ankle', 'R_ankle', 'L_big_toe',
            #               'R_big_toe', 'L_small_toe', 'R_small_toe', 'L_heel', 'R_heel', 'L_hand_thumb', 'R_hand_thumb',
            #               'L_hand_index', 'R_hand_index', 'L_hand_middle', 'R_hand_middle', 'L_hand_pinky',
            #               'R_hand_pinky']


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
    s.req_exercise = "help_function"
    time.sleep(2)
    # Create all components
    s.camera = Realsense()
    s.number_of_repetitions_in_training=0


    # Start all threads
    s.camera.start()






