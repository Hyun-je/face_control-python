import time
from array import array
from typing import Tuple, Dict, Any

import liblo as OSC
from ola.ClientWrapper import ClientWrapper

# === 조명 / DMX 세팅 ===
UNIVERSE = 1234
MAX_FACES = 4  # 최대 동시 제어할 조명(얼굴) 수

# 각도 범위
PAN_MIN, PAN_MAX = 0, 540
TILT_MIN, TILT_MAX = 0, 270

# 초기 위치
DEFAULT_PAN = 270.0
DEFAULT_TILT = 90.0

# OSC 타임아웃(초) - 이 시간 동안 새 데이터가 없으면 조명이 기본 위치로 돌아감
OSC_TIMEOUT_SEC = 5.0
# =======================

# 얼굴별 데이터 저장: { face_index: {'pan': float, 'tilt': float, 'last_update': float} }
faces_data: Dict[int, Dict[str, Any]] = {}


def get_dmx_channels_for_face(face_index: int) -> Tuple[int, int] | None:
    """
    얼굴 인덱스에 해당하는 DMX pan/tilt 채널을 반환합니다.
    조명 설정에 맞게 이 함수를 수정하여 채널 매핑을 변경할 수 있습니다.

    :param face_index: 얼굴 인덱스 (0부터 시작)
    :return: (pan_channel, tilt_channel) 튜플 또는 유효하지 않은 인덱스의 경우 None
    """
    if not 0 <= face_index < MAX_FACES:
        return None

    # 예시: 각 조명이 10개의 채널 블록을 차지한다고 가정
    # 얼굴 0 -> pan: 0, tilt: 2
    # 얼굴 1 -> pan: 10, tilt: 12
    base_channel = face_index * 10
    pan_channel = base_channel
    tilt_channel = base_channel + 2
    return pan_channel, tilt_channel


def angle_to_dmx(angle, min_angle, max_angle):
    """각도를 0-255 범위의 DMX 값으로 변환합니다."""
    angle = max(min_angle, min(angle, max_angle))  # 범위를 벗어나면 클리핑
    return int(((angle - min_angle) / (max_angle - min_angle)) * 255)


class PanTiltOSCServer(OSC.ServerThread):
    def __init__(self, port):
        super().__init__(port)
        self.add_method("/pan_tilt", None, self.pan_tilt_handler)
        print(f"OSC 서버 시작: 0.0.0.0:{port} (주소: /pan_tilt, 포맷: [face_idx, pan, tilt])")


    def pan_tilt_handler(self, path, args, types, src):
        """OSC 메시지 수신 핸들러"""
        if len(args) < 3:
            print(f"[OSC] 인자 부족 (3개 필요), 값 무시. 수신: {args}")
            return

        try:
            face_index, pan_val, tilt_val = int(args[0]), float(args[1]), float(args[2])
        except (ValueError, TypeError):
            print(f"[OSC] 잘못된 인자 타입 수신: {args}")
            return

        if 0 <= face_index < MAX_FACES:
            faces_data[face_index] = {
                "pan": pan_val,
                "tilt": tilt_val,
                "last_update": time.time(),
            }
            # print(f"[OSC] 수신 - Face {face_index}: Pan={pan_val:.1f}, Tilt={tilt_val:.1f}")
        else:
            print(f"[OSC] 범위를 벗어난 얼굴 인덱스 수신: {face_index}")


def dmx_send_callback(state):
    """DMX 전송 완료 후 호출될 콜백 (필요시 사용)"""
    if not state.Ok():
        print("DMX 전송 실패")


def main():
    osc_server = PanTiltOSCServer(5000)
    osc_server.start()

    wrapper = ClientWrapper()
    client = wrapper.Client()

    try:
        while True:
            now = time.time()
            data = array("B", [0] * 512)
            status_parts = []

            for i in range(MAX_FACES):
                channels = get_dmx_channels_for_face(i)
                if not channels:
                    continue

                pan_ch, tilt_ch = channels
                face_info = faces_data.get(i)

                # 타임아웃 확인
                if face_info and (now - face_info["last_update"] > OSC_TIMEOUT_SEC):
                    # print(f"Face {i} 타임아웃. 기본 위치로 재설정.")
                    del faces_data[i]
                    face_info = None

                if face_info:  # 활성 얼굴
                    target_pan = face_info["pan"]
                    target_tilt = face_info["tilt"]
                    status_char = "▶"  # 활성
                else:  # 비활성 또는 타임아웃된 얼굴
                    target_pan = DEFAULT_PAN
                    target_tilt = DEFAULT_TILT
                    status_char = "■"  # 비활성/기본값

                pan_dmx = angle_to_dmx(target_pan, PAN_MIN, PAN_MAX)
                tilt_dmx = angle_to_dmx(target_tilt, TILT_MIN, TILT_MAX)

                if pan_ch < 512 and tilt_ch < 512:
                    data[pan_ch] = pan_dmx
                    data[tilt_ch] = tilt_dmx

                status_parts.append(f"F{i}:{status_char}")

            print(f"\rDMX Status: [{' '.join(status_parts)}]", end="")

            client.SendDmx(UNIVERSE, data, dmx_send_callback)
            wrapper.Run()

            time.sleep(0.02)  # 약 50Hz

    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        print("\nPan/Tilt를 0으로 초기화하고 종료합니다.")
        data = array("B", [0] * 512)

        pan_dmx_at_zero = angle_to_dmx(0, PAN_MIN, PAN_MAX)
        tilt_dmx_at_zero = angle_to_dmx(0, TILT_MIN, TILT_MAX)

        for i in range(MAX_FACES):
            channels = get_dmx_channels_for_face(i)
            if not channels:
                continue

            pan_ch, tilt_ch = channels
            if pan_ch < 512 and tilt_ch < 512:
                data[pan_ch] = pan_dmx_at_zero
                data[tilt_ch] = tilt_dmx_at_zero

        client.SendDmx(UNIVERSE, data)
        wrapper.Run()

        osc_server.close()
        print("OSC 서버 종료.")


if __name__ == "__main__":
    main()
