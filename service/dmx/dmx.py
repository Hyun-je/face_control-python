import time
import threading
from pythonosc import dispatcher
from pythonosc import osc_server
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

# OSC 타임아웃(초)
OSC_TIMEOUT_SEC = 60.0
# =======================

# OSC로 받을 전역 변수 (초기 위치로 시작)
pan  = DEFAULT_PAN
tilt = DEFAULT_TILT

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

def pan_tilt_handler(address, *args):
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

    # --- OSC 서버 설정 ---
    disp = dispatcher.Dispatcher()
    disp.map("/pan_tilt_*", pan_tilt_handler)

    osc_ip = "0.0.0.0"  # 모든 IP에서 수신
    osc_port = 5000
    server = osc_server.ThreadingOSCUDPServer((osc_ip, osc_port), disp)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print(f"OSC 서버 시작: udp://{osc_ip}:{osc_port} (주소: /pan_tilt)")
    # --------------------

    wrapper = ClientWrapper()
    client = wrapper.Client()

    try:
        while True:
            now = time.time()

            # 🔹 1분 이상 OSC 수신이 없으면 초기 위치로 복귀
            if now - last_osc_time > OSC_TIMEOUT_SEC:
                target_pan  = DEFAULT_PAN
                target_tilt = DEFAULT_TILT
                timeout_state = "(타임아웃 → 초기 위치)"
            else:
                target_pan  = pan
                target_tilt = tilt
                timeout_state = ""

            data = array('B', [0] * 512)

            pan_dmx  = angle_to_dmx(target_pan, PAN_MIN, PAN_MAX)
            tilt_dmx = angle_to_dmx(target_tilt, TILT_MIN, TILT_MAX)

            data[PAN_CH]  = pan_dmx
            data[TILT_CH] = tilt_dmx

            print(f"DMX 전송 {timeout_state} "
                  f"- pan: {target_pan:.1f} -> CH{PAN_CH+1}={pan_dmx}, "
                  f"tilt: {target_tilt:.1f} -> CH{TILT_CH+1}={tilt_dmx}")

            client.SendDmx(UNIVERSE, data, dmx_sent)
            wrapper.Run()

            time.sleep(0.02)  # 약 50Hz

    except KeyboardInterrupt:
        print("종료합니다.")
        server.shutdown()
        server_thread.join()
    finally:
        print("프로그램을 종료하기 전에 Pan/Tilt를 0으로 리셋합니다.")

        # Final DMX send needs its own callback
        def dmx_sent_on_exit(state):
            wrapper.Stop()

        data = array('B', [0] * 512)
        data[PAN_CH] = 0
        data[TILT_CH] = 0

        print(f"DMX 리셋 신호 전송: CH{PAN_CH}=0, CH{TILT_CH}=0")
        client.SendDmx(UNIVERSE, data, dmx_sent_on_exit)
        wrapper.Run()
        print("리셋 완료. 프로그램이 종료되었습니다.")

if __name__ == '__main__':
    main()
