import threading
import time
import numpy as np
import cv2
import pyzed.sl as sl
import tkinter as tk
from tkinter import ttk
import Settings as s  # Controlled shutdown
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

class PyZedWrapper(threading.Thread):
    def __init__(self, update_gui_callback):
        """Initialize the ZED Camera Thread."""
        super().__init__()
        self.zed = sl.Camera()
        self.lock = threading.Lock()
        self.latest_keypoints = None
        self.running = True
        self.update_gui_callback = update_gui_callback  # GUI update callback

    def stop(self):
        """Stop the thread safely."""
        self.running = False
        s.finish_program = True

    def get_latest_keypoints(self):
        """Retrieve the latest detected 3D keypoints safely."""
        with self.lock:
            return self.latest_keypoints.copy() if self.latest_keypoints is not None else None

    def run(self):
        """Main thread function to capture frames and detect 3D keypoints."""
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # 3D-Friendly coordinate system
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_units = sl.UNIT.MILLIMETER
        init.camera_fps = 30

        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            self.zed.close()
            exit(1)

        body_params = sl.BodyTrackingParameters()
        body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM
        body_params.enable_tracking = True
        body_params.enable_body_fitting = True
        body_params.body_format = sl.BODY_FORMAT.BODY_18
        self.zed.enable_body_tracking(body_params)

        runtime = sl.RuntimeParameters()
        bodies = sl.Bodies()
        selected_keypoints = list(range(18))  # Use all 18 keypoints

        while self.zed.is_opened() and self.running:
            if self.zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_bodies(bodies, sl.BodyTrackingRuntimeParameters())
                body_list = bodies.body_list

                keypoints_list = []
                for body in body_list:
                    if body.keypoint is not None:
                        keypoints = np.array(body.keypoint, dtype=float)  # Get 3D keypoints
                        for idx in selected_keypoints:
                            if len(keypoints) > idx:
                                kp = keypoints[idx]
                                keypoints_list.append(kp if kp[0] != 0 else [np.nan, np.nan, np.nan])

                with self.lock:
                    self.latest_keypoints = keypoints_list

                self.update_gui_callback()  # Update GUI with new keypoints

        self.zed.close()
        print("Camera closed")


class PatientGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Patient Progress Monitor")

        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(root, length=400, mode='determinate', variable=self.progress_var)
        self.progress_bar.pack(pady=10)

        # Feedback Label
        self.feedback_label = tk.Label(root, text="Adjust your position", font=("Arial", 14))
        self.feedback_label.pack(pady=5)

        # 3D Figure Setup
        self.figure = plt.figure(figsize=(5, 5))
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(pady=10)

        # Start ZED camera thread
        self.zed_thread = PyZedWrapper(self.update_gui)
        self.zed_thread.start()

        self.update_gui()

    def draw_3d_stick_figure(self, keypoints):
        """Draw a simple 3D stick figure based on 3D keypoints."""
        self.ax.clear()
        self.ax.set_xlim(-500, 500)
        self.ax.set_ylim(-500, 500)
        self.ax.set_zlim(0, 1500)  # Assuming height is 1.5m
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.view_init(elev=20, azim=30)  # Set good viewing angle

        if keypoints is None:
            return

        # Define key joint connections (Skeleton Structure)
        body_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Right arm
            (0, 5), (5, 6), (6, 7), (7, 8),  # Left arm
            (0, 9), (9, 10), (10, 11), (11, 12),  # Right leg
            (0, 13), (13, 14), (14, 15), (15, 16)  # Left leg
        ]

        # Draw skeleton in 3D
        for joint1, joint2 in body_connections:
            if all(not np.isnan(kp[0]) for kp in [keypoints[joint1], keypoints[joint2]]):
                self.ax.plot(
                    [keypoints[joint1][0], keypoints[joint2][0]],
                    [keypoints[joint1][1], keypoints[joint2][1]],
                    [keypoints[joint1][2], keypoints[joint2][2]], 'ro-')

        self.canvas.draw()

    def update_gui(self):
        """Update progress bar and visual feedback based on keypoints."""
        keypoints = self.zed_thread.get_latest_keypoints()
        if keypoints is not None:
            self.draw_3d_stick_figure(keypoints)

            # Example logic for feedback
            right_hand_z = keypoints[3][2] if not np.isnan(keypoints[3][2]) else None
            left_hand_z = keypoints[7][2] if not np.isnan(keypoints[7][2]) else None

            if right_hand_z and right_hand_z < 600:
                feedback = "Move your right hand forward!"
            elif left_hand_z and left_hand_z < 600:
                feedback = "Move your left hand forward!"
            else:
                feedback = "Good position!"

            self.feedback_label.config(text=feedback)

            # Simulated progress bar based on hand distance
            progress = (1000 - right_hand_z) / 10 if right_hand_z else 50
            self.progress_var.set(max(0, min(100, progress)))

        self.root.after(100, self.update_gui)  # Schedule next update

    def stop(self):
        """Stop camera thread when closing the GUI."""
        self.zed_thread.stop()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    gui = PatientGUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.stop)
    root.mainloop()
