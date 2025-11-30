# dmx_control.py
# This script now captures webcam video, detects face orientation,
# and sends DMX pan/tilt commands via OSC.
# It is based on the logic from main.py and the original dmx_control.py.

from __future__ import annotations

import time
import argparse
import sys
from typing import Tuple

import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
from scipy.spatial import distance as dist
from scipy.spatial.transform import Rotation as R
import yaml

# Ensure FaceAnalyzer is available. You may need to install it:
# pip install FaceAnalyzer
from FaceAnalyzer import FaceAnalyzer, Face

WINDOW_NAME = "DMX Face Control"


def calculate_ear(eye: np.ndarray) -> float:
    """Calculate the Eye Aspect Ratio for a single eye."""
    # Vertical distances
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    # Horizontal distance
    c = dist.euclidean(eye[0], eye[3])
    # Compute EAR, avoiding division by zero
    if c == 0:
        return 1.0  # Should not happen with real landmarks
    ear = (a + b) / (2.0 * c)
    return float(ear)


def calculate_mar(mouth: np.ndarray) -> float:
    """Calculate the Mouth Aspect Ratio for the mouth."""
    # Vertical distance between inner lips (top-bottom landmarks)
    a = dist.euclidean(mouth[1], mouth[2])
    # Horizontal distance between mouth corners (left-right landmarks)
    b = dist.euclidean(mouth[0], mouth[3])
    # Compute MAR, avoiding division by zero
    if b == 0:
        return 0.0  # Should not happen with real landmarks
    mar = a / b
    return float(mar)


def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
    corrected_pitch = pitch #- pitch_correction_angle_deg

    return corrected_yaw, corrected_pitch, roll


def _calculate_pitch_from_landmarks(
    landmarks: np.ndarray, frame_shape: Tuple[int, int]
) -> float | None:
    """Calculates head pitch based on the relative positions of facial landmarks."""
    try:
        # MediaPipe landmark indices
        NOSE_TIP = 1
        NOSE_BRIDGE = 168  # A point on the nose bridge, between the eyes
        CHIN = 152

        # Get pixel coordinates for landmarks
        nose_tip_pt = landmarks[NOSE_TIP]
        nose_bridge_pt = landmarks[NOSE_BRIDGE]
        chin_pt = landmarks[CHIN]

        # Calculate vertical distances
        # Positive y is downwards
        dist_nose_bridge_to_tip = nose_tip_pt[1] - nose_bridge_pt[1]
        dist_tip_to_chin = chin_pt[1] - nose_tip_pt[1]

        # Avoid division by zero
        if dist_tip_to_chin == 0:
            return 0.0

        # The ratio indicates how much the nose is pointing up or down
        # A neutral pose has a ratio around 0.6-0.8
        ratio = dist_nose_bridge_to_tip / dist_tip_to_chin

        # Map the ratio to a pitch angle. This is an empirical mapping and may need tuning.
        # We map a neutral ratio (e.g., 0.75) to 0 degrees.
        # Values lower than neutral mean head is tilted up (negative pitch).
        # Values higher than neutral mean head is tilted down (positive pitch).
        pitch_sensitivity = -200  # Multiplier to scale the ratio to a degree-like value
        neutral_ratio = 0.75
        pitch = (ratio - neutral_ratio) * pitch_sensitivity

        # Clamp the pitch to a reasonable range, e.g., [-90, 90]
        pitch = np.clip(pitch, -90.0, 90.0)

        return float(pitch)

    except (IndexError, TypeError):
        return None


def extract_head_orientation(
    face: Face, frame_shape: Tuple[int, int]
) -> Tuple[float, float, float, float, float] | None:
    """
    Return yaw, pitch, roll.
    - Yaw and Roll are derived from FaceAnalyzer's head posture.
    - Pitch is calculated independently using facial landmarks.
    """
    try:
        # Get Yaw and Roll from the 3D model (more stable for these axes)
        position, orientation = face.get_head_posture()
        if position is None or orientation is None:
            return None
        yaw, _, roll = correct_orientation_for_perspective(position, orientation)

        # Calculate Pitch from landmarks for more direct control
        landmarks_normalized = face.landmarks
        if not landmarks_normalized:
            return None

        # Convert normalized landmarks to pixel coordinates
        frame_height, frame_width, _ = frame_shape
        all_landmarks = np.array(
            [(lm.x * frame_width, lm.y * frame_height) for lm in landmarks_normalized.landmark],
            dtype=np.float32,
        )

        pitch = _calculate_pitch_from_landmarks(all_landmarks, frame_shape)
        if pitch is None:
            # Fallback to original pitch if landmark calculation fails
            _, pitch, _ = correct_orientation_for_perspective(position, orientation)
        pitch -= 30

    except Exception:
        return None

    x1, y1, x2, y2 = face.bounding_box
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2

    return float(x), float(y), float(yaw), float(pitch), float(roll)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DMX Face Control: Tracks faces and sends OSC commands."
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=None,
        help="Path to video file (optional). If not provided, uses webcam.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable GUI display for debugging."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="service/face/config.yaml",
        help="Path to the configuration YAML file.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    config = load_config(args.config)

    # --- OSC and DMX Configuration ---
    # Define target IPs and ports for up to 3 faces
    targets = config["dmx_osc"]
    clients = [SimpleUDPClient(target["ip"], target["port"]) for target in targets]
    print("OSC senders ready:")
    for i, target in enumerate(targets):
        print(
            f"  Client {i} -> {target['ip']}:{target['port']} (Address: /pan_tilt_N, /blink_N, /mouth_open_N)"
        )

    # OSC client for Max/DSP on the same computer
    max_ip = config["max_osc"]["ip"]
    max_port = config["max_osc"]["port"]
    max_client = SimpleUDPClient(max_ip, max_port)
    print(f"OSC sender for Max/MSP ready -> {max_ip}:{max_port} (Address: /pan_N, /tilt_N, /blink_N, /mouth_open_N)")

    # --- Blink Detection Configuration ---
    EYE_AR_THRESH = 0.22  # Threshold for eye aspect ratio to trigger a blink
    EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames the eye must be below threshold
    blink_counter = 0
    total_blinks = 0

    # --- Mouth Open Detection Configuration ---
    MOUTH_AR_THRESH = 0.4  # Threshold for mouth aspect ratio to trigger open
    MOUTH_AR_CONSEC_FRAMES = 3  # Number of consecutive frames mouth must be open
    mouth_open_counter = 0
    total_mouth_opens = 0

    # --- Face Tracking Configuration ---
    cap = configure_capture(args.video_path)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    analyzer = FaceAnalyzer(max_nb_faces=1, image_shape=(frame_width, frame_height))
    
    if args.debug:
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

            if face_count > 0 and faces:
                # Process up to 3 faces
                for index, face in enumerate(faces[:3]):
                    if not face.ready:
                        continue

                    # Select the client based on the face index
                    if index >= len(clients):
                        continue
                    client = clients[index]

                    orientation_data = extract_head_orientation(face, frame.shape)

                    if orientation_data:
                        _, _, yaw, pitch, roll = orientation_data

                        # --- Map Yaw/Pitch to Pan/Tilt ---
                        # Pan mapping: yaw [-90, 90] -> pan [180, 360]
                        pan = yaw * yaw * yaw / 1000.0 + 270.0
                        pan = max(90.0, min(450.0, pan))

                        # Tilt mapping: pitch [-90, 90] -> tilt [0, 180]
                        # Inverted so that head up = light tilts down
                        if pitch < 0:
                            tilt = pitch * pitch * pitch / 2000.0 + 90.0
                        else:
                            tilt = pitch * pitch * pitch / 2000.0 + 90.0
                        tilt = max(0.0, min(180.0, tilt))

                        # --- Send OSC Message ---
                        client.send_message(f"/pan_tilt_{index}", [pan, tilt])
                        max_client.send_message(f"/pan_{index}", pan)
                        max_client.send_message(f"/tilt_{index}", tilt)

                        # --- Blink and Mouth Detection ---
                        try:
                            landmarks_normalized = face.landmarks
                            frame_height, frame_width, _ = frame.shape

                            # Convert NormalizedLandmarkList to a NumPy array of pixel coordinates
                            all_landmarks = np.array(
                                [
                                    (lm.x * frame_width, lm.y * frame_height)
                                    for lm in landmarks_normalized.landmark
                                ],
                                dtype=np.float32,
                            )

                            # --- Blink Detection ---
                            # MediaPipe 6-point eye landmark indices for EAR calculation
                            LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
                            RIGHT_EYE_IDXS = [263, 387, 385, 362, 380, 373]

                            left_eye = all_landmarks[LEFT_EYE_IDXS]
                            right_eye = all_landmarks[RIGHT_EYE_IDXS]

                            left_ear = calculate_ear(left_eye)
                            right_ear = calculate_ear(right_eye)

                            ear = (left_ear + right_ear) / 2.0

                            # NOTE: For now, blink_counter and total_blinks are shared across all faces.
                            # A more robust solution would involve per-face blink tracking.
                            if ear < EYE_AR_THRESH:
                                blink_counter += 1
                            else:
                                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                                    total_blinks += 1
                                    client.send_message(f"/blink_{index}", 1)
                                    max_client.send_message(f"/blink_{index}", 1)
                                    print(f"Face {index} Blink detected! Total: {total_blinks}")
                                blink_counter = 0

                            # --- Mouth Open Detection ---
                            # MediaPipe 4-point mouth landmark indices for MAR calculation
                            MOUTH_IDXS = [61, 13, 14, 291]  # Left corner, top lip, bottom lip, right corner
                            mouth = all_landmarks[MOUTH_IDXS]
                            mar = calculate_mar(mouth)

                            # Same note applies to mouth open counter.
                            if mar > MOUTH_AR_THRESH:
                                mouth_open_counter += 1
                            else:
                                if mouth_open_counter >= MOUTH_AR_CONSEC_FRAMES:
                                    total_mouth_opens += 1
                                    client.send_message(f"/mouth_open_{index}", 1)
                                    max_client.send_message(f"/mouth_open_{index}", 1)
                                    print(f"Face {index} Mouth Open detected! Total: {total_mouth_opens}")
                                mouth_open_counter = 0

                            if args.debug:
                                # Add info text below the bounding box
                                (x1, _, _, y2) = face.bounding_box
                                info_text_blink = f"Blinks: {total_blinks} EAR: {ear:.2f}"
                                cv2.putText(
                                    frame,
                                    info_text_blink,
                                    (int(x1), int(y2) + 15),  # Position below the bounding box
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    1,
                                    cv2.LINE_AA,
                                )

                                info_text_mouth = f"Opens: {total_mouth_opens} MAR: {mar:.2f}"
                                cv2.putText(
                                    frame,
                                    info_text_mouth,
                                    (int(x1), int(y2) + 35),  # Position below the blink text
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    1,
                                    cv2.LINE_AA,
                                )

                        except (AttributeError, IndexError):
                            # Landmarks are not available
                            pass

                        if args.debug:
                            # --- Display Info on Frame ---
                            face.draw_bounding_box(frame, color=(0, 255, 0))
                            face.draw_landmarks(
                                frame, radius=1, thickness=2, color=(255, 255, 255)
                            )
                            info_text = (
                                f"Face {index} Yaw:{yaw:5.1f} Pitch:{pitch:5.1f} | Pan:{pan:5.1f} Tilt:{tilt:5.1f}"
                            )
                            (x1, y1, _, _) = face.bounding_box
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

            if args.debug:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(33) & 0xFF == ord("q"):
                    print("\n'q' pressed. Shutting down.")
                    break
            else:
                time.sleep(33.0 / 1000.0)

    except KeyboardInterrupt:
        print("\nShutdown requested.")
    finally:
        for index in range(3):
            client.send_message(f"/pan_tilt_{index}", [270, 90])
        cap.release()
        if args.debug:
            cv2.destroyAllWindows()
        print("Resources released.")


if __name__ == "__main__":
    main()