# dmx_control.py
# This script now captures webcam video, detects face orientation,
# and sends DMX pan/tilt commands via OSC.
# It is based on the logic from main.py and the original dmx_control.py.

from __future__ import annotations

import sys
from typing import Tuple

import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
from scipy.spatial.transform import Rotation as R

# Ensure FaceAnalyzer is available. You may need to install it:
# pip install FaceAnalyzer
from FaceAnalyzer import FaceAnalyzer, Face

WINDOW_NAME = "DMX Face Control"


def configure_capture(video_path: str | None = None) -> cv2.VideoCapture:
    """Create a capture object from a video file or webcam."""
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
    else:
        # Use AVFoundation for macOS for better performance
        capture_api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(0, capture_api)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)  # Fallback to default
        if not cap.isOpened():
            raise RuntimeError("Failed to open the default camera")
    return cap


def correct_orientation_for_perspective(
    face_pos: np.ndarray,
    face_ori: np.ndarray,
) -> Tuple[float, float, float]:
    """Corrects for perspective distortion in yaw and pitch."""
    tvec_x_original = -face_pos[0]
    tvec_y_original = -face_pos[1]
    tvec_z_original = face_pos[2]

    yaw_correction_angle_rad = np.arctan2(tvec_x_original, tvec_z_original)
    yaw_correction_angle_deg = np.degrees(yaw_correction_angle_rad)

    pitch_correction_angle_rad = np.arctan2(tvec_y_original, tvec_z_original)
    pitch_correction_angle_deg = np.degrees(pitch_correction_angle_rad)

    rot = R.from_rotvec(face_ori.flatten())
    euler_angles = rot.as_euler("yxz", degrees=True)
    yaw, pitch, roll = euler_angles[0], euler_angles[1], euler_angles[2]

    corrected_yaw = yaw - yaw_correction_angle_deg
    corrected_pitch = pitch + pitch_correction_angle_deg

    return corrected_yaw, corrected_pitch, roll


def extract_head_orientation(
    face: Face,
) -> Tuple[float, float, float, float, float] | None:
    """Return yaw, pitch, roll derived from FaceAnalyzer's head posture."""
    try:
        position, orientation = face.get_head_posture()
        if position is None or orientation is None:
            return None
        yaw, pitch, roll = correct_orientation_for_perspective(position, orientation)
    except Exception:
        return None

    x1, y1, x2, y2 = face.bounding_box
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2

    return float(x), float(y), float(yaw), float(pitch), float(roll)

def main():
    # --- OSC and DMX Configuration ---
    target_ip = "192.168.10.38"  # IP of the receiving device
    target_port = 5000  # OSC port
    client = SimpleUDPClient(target_ip, target_port)
    print(f"OSC sender ready -> {target_ip}:{target_port} (Address: /pan_tilt)")

    # OSC client for Max/DSP on the same computer
    max_ip = "127.0.0.1"  # Localhost
    max_port = 8000  # Common port for Max/MSP, change if needed
    max_client = SimpleUDPClient(max_ip, max_port)
    print(f"OSC sender for Max/MSP ready -> {max_ip}:{max_port} (Address: /pan_tilt)")

    # --- Face Tracking Configuration ---
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    cap = configure_capture(video_path)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    analyzer = FaceAnalyzer(max_nb_faces=1, image_shape=(frame_width, frame_height))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("Starting face tracking for DMX control. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed; exiting.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            analyzer.process(rgb_frame)

            faces = getattr(analyzer, "faces", [])
            face_count = getattr(analyzer, "nb_faces", len(faces))

            if face_count > 0 and faces and faces[0].ready:
                orientation_data = extract_head_orientation(faces[0])

                if orientation_data:
                    _, _, yaw, pitch, roll = orientation_data

                    # --- Map Yaw/Pitch to Pan/Tilt ---
                    # Pan mapping: yaw [-90, 90] -> pan [180, 360]
                    pan = yaw * 1.2 + 270.0
                    pan = max(180.0, min(360.0, pan))

                    # Tilt mapping: pitch [-90, 90] -> tilt [0, 180]
                    # Inverted so that head up = light tilts down
                    if pitch < 0:
                        tilt = pitch * 3 + 90.0
                    else:
                        tilt = pitch * 2 + 90.0
                    tilt = max(0.0, min(180.0, tilt))
                    
                    # --- Send OSC Message ---
                    client.send_message("/pan_tilt", [pan, tilt])
                    max_client.send_message("/pan", pan)
                    max_client.send_message("/tilt", tilt)
                    
                    # --- Display Info on Frame ---
                    faces[0].draw_bounding_box(frame, color=(0, 255, 0))
                    faces[0].draw_landmarks(frame, radius=1, thickness=2, color=(255, 255, 255))
                    info_text = f"Yaw:{yaw:5.1f} Pitch:{pitch:5.1f} | Pan:{pan:5.1f} Tilt:{tilt:5.1f}"
                    (x1, y1, _, _) = faces[0].bounding_box
                    cv2.putText(
                        frame,
                        info_text,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
            else:
                 # If no face is detected, you might want to send a default position, e.g., center
                 # client.send_message("/pan_tilt", [270.0, 90.0])
                 pass


            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n'q' pressed. Shutting down.")
                break
    except KeyboardInterrupt:
        print("\nShutdown requested.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")


if __name__ == "__main__":
    main()