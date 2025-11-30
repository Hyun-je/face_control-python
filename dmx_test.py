import time
import liblo as OSC
from ola.ClientWrapper import ClientWrapper
from array import array

# === 조명 / DMX 세팅 ===
UNIVERSE = 0

# DMX 채널 번호 (0부터 시작)
PAN_CH   = 0   # DMX 채널 1
TILT_CH  = 1   # DMX 채널 2 (조명 매뉴얼 보고 필요시 수정)

# 각도 범위
PAN_MIN, PAN_MAX   = 0, 540
TILT_MIN, TILT_MAX = 0, 720

# 초기 위치
DEFAULT_PAN  = 270.0
DEFAULT_TILT = 90.0

# 마지막 OSC 수신 시각
last_osc_time = time.time()

def angle_to_dmx(angle, min_angle, max_angle):
    # 범위를 벗어나면 클리핑
    if angle < min_angle:
        angle = min_angle
    if angle > max_angle:
        angle = max_angle
    # 선형 매핑
    return int((angle - min_angle) / (max_angle - min_angle) * 255)

class PanTiltOSCServer(OSC.ServerThread):
    def __init__(self, port):
        OSC.ServerThread.__init__(self, port)
        # 타입을 None으로 두면 어떤 타입이 와도 처리
        self.add_method("/pan_tilt", None, self.pan_tilt_handler)

    def pan_tilt_handler(self, path, args, types, src):
        global pan, tilt, last_osc_time
        if len(args) >= 2:
            pan_val, tilt_val = args[0], args[1]
            pan = float(pan_val)
            tilt = float(tilt_val)
            last_osc_time = time.time()   # 🔹 마지막 수신 시각 갱신
            print(f"[OSC] 수신 - pan: {pan}, tilt: {tilt}")
        else:
            print("[OSC] 인자 부족, 값 무시")

def main():

    def dmx_sent(state):
        wrapper.Stop()

    osc_port = 5000
    osc_server = PanTiltOSCServer(osc_port)
    osc_server.start()
    print(f"OSC 서버 시작: udp://192.168.10.38:{osc_port} (주소: /pan_tilt)")

    wrapper = ClientWrapper()
    client = wrapper.Client()

    try:
        while True:
            pan_str = input(f"Face Pan 값 ({PAN_MIN}-{PAN_MAX}): ")
            if pan_str.lower() == 'q':
                break
            target_pan = float(pan_str)

            tilt_str = input(f"Face Tilt 값 ({TILT_MIN}-{TILT_MAX}): ")
            if tilt_str.lower() == 'q':
                break
            target_tilt = float(tilt_str)

            data = array('B', [0] * 512)

            pan_dmx  = angle_to_dmx(target_pan, PAN_MIN, PAN_MAX)
            tilt_dmx = angle_to_dmx(target_tilt, TILT_MIN, TILT_MAX)

            data[PAN_CH]  = pan_dmx
            data[TILT_CH] = tilt_dmx

            print(f"DMX 전송 "
                  f"- pan: {target_pan:.1f} -> CH{PAN_CH}={pan_dmx}, "
                  f"tilt: {target_tilt:.1f} -> CH{TILT_CH}={tilt_dmx}")

            client.SendDmx(UNIVERSE, data, dmx_sent)
            wrapper.Run()

    except KeyboardInterrupt:
        print("종료합니다.")

if __name__ == '__main__':
    main()
