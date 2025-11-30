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

def angle_to_dmx(angle, min_angle, max_angle):
    # 범위를 벗어나면 클리핑
    if angle < min_angle:
        angle = min_angle
    if angle > max_angle:
        angle = max_angle
    # 선형 매핑
    return int((angle - min_angle) / (max_angle - min_angle) * 255)


def main():

    def dmx_sent(state):
        wrapper.Stop()

    wrapper = ClientWrapper()
    client = wrapper.Client()

    try:
        while True:
            try:
                pan_str = input(f"Face Pan 값 ({PAN_MIN}-{PAN_MAX}): ")
                if pan_str.lower() == 'q':
                    break
                target_pan = float(pan_str)

                tilt_str = input(f"Face Tilt 값 ({TILT_MIN}-{TILT_MAX}): ")
                if tilt_str.lower() == 'q':
                    break
                target_tilt = float(tilt_str)
            except ValueError:
                print("잘못된 입력입니다. 숫자를 입력하거나 'q'로 종료하세요.")
                continue

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
        print("\n키보드 입력으로 종료 절차를 시작합니다.")
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
