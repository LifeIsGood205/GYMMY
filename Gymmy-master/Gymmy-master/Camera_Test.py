import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

def initialize_camera_and_pose():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)


    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    return pipeline, pose, mp_pose

def get_skeleton_data(pipeline, pose):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None, None

    image = np.asanyarray(color_frame.get_data())
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)

    if not results.pose_landmarks:
        return image, None

    h, w, _ = image.shape
    joints = {}
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        x, y, z = int(lm.x * w), int(lm.y * h), int(lm.z*100)
        joints[idx] = (x, y, z)
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    # Draw limbs by connecting joints based on mediapipe connections
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx in joints and end_idx in joints:
            start_point = joints[start_idx][:2]
            end_point = joints[end_idx][:2]
            cv2.line(image, start_point, end_point, (255, 255, 255), 2)

    return image, joints

# Example usage:
if __name__ == "__main__":
    pipeline, pose, mp_pose = initialize_camera_and_pose()

    try:
        while True:
            image, joints = get_skeleton_data(pipeline, pose)
            if image is None:
                continue

            if joints:
                for joint_id, coord in joints.items():
                    print(f"Joint {joint_id}: {coord}")

            cv2.imshow("Skeleton", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        pose.close()
        cv2.destroyAllWindows()
