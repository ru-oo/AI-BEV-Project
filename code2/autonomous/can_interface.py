"""
can_interface.py - STM32 CAN 통신 인터페이스
=============================================
PC → STM32 CAN 통신으로 자율주행 명령 전송

CAN 메시지 프로토콜:
  ID 0x100 | 8 bytes | 조향·속도 제어
    Byte 0-1 : steering  (int16, -1000~+1000, ×0.001 = -1.0~1.0)
    Byte 2-3 : speed     (uint16, 0~1000,     ×0.001 = 0.0~1.0)
    Byte 4   : mode      (uint8, 0=STOP, 1=AUTO, 2=MANUAL)
    Byte 5-7 : reserved

  ID 0x101 | 2 bytes | 상태 확인 (수신)
    Byte 0 : STM32 ready flag
    Byte 1 : error code

필요 라이브러리: pip install python-can
USB-CAN 어댑터: slcan (CANable, etc.) 또는 socketcan (Linux)
"""

import struct
import time
import threading


class STM32CANInterface:
    """
    STM32 CAN 통신 래퍼
    연결 실패 시 자동으로 시뮬레이션 모드 전환
    """

    # CAN ID
    ID_CONTROL = 0x100
    ID_STATUS  = 0x101

    def __init__(self,
                 channel: str  = 'COM3',
                 bustype: str  = 'slcan',
                 bitrate: int  = 500_000,
                 verbose: bool = False):
        """
        Parameters
        ----------
        channel  : CAN 포트 (Windows: 'COM3', Linux: 'can0')
        bustype  : 'slcan' (USB-CAN) | 'socketcan' (Linux 내장)
        bitrate  : CAN 속도 (기본 500kbps)
        verbose  : 상세 로그 출력
        """
        self.verbose   = verbose
        self.connected = False
        self._lock     = threading.Lock()
        self._last_cmd = (0.0, 0.0, 0)   # steer, speed, mode

        try:
            import can
            self.bus = can.interface.Bus(
                channel=channel,
                bustype=bustype,
                bitrate=bitrate
            )
            self.connected = True
            print(f"  ✅ CAN 연결 성공: {channel} ({bustype}, {bitrate//1000}kbps)")

            # 수신 스레드 시작
            self._rx_thread = threading.Thread(
                target=self._rx_loop, daemon=True)
            self._rx_thread.start()

        except Exception as e:
            print(f"  ⚠️  CAN 연결 실패: {e}")
            print("  📺 시뮬레이션 모드로 실행합니다.")

    # ── 제어 명령 전송 ────────────────────────
    def send_control(self,
                     steering: float,
                     speed:    float,
                     mode:     int = 1) -> bool:
        """
        Parameters
        ----------
        steering : -1.0(좌) ~ +1.0(우)
        speed    :  0.0     ~ +1.0
        mode     :  0=STOP, 1=AUTO, 2=MANUAL

        Returns True if sent successfully
        """
        steering = float(max(-1.0, min(1.0, steering)))
        speed    = float(max(0.0,  min(1.0, speed)))

        steer_raw = int(steering * 1000)
        speed_raw = int(speed    * 1000)

        data = struct.pack('>hHBxxx', steer_raw, speed_raw, mode)  # 8 bytes

        self._last_cmd = (steering, speed, mode)

        if self.verbose:
            print(f"  [CAN TX] steer={steering:+.3f}  "
                  f"speed={speed:.3f}  mode={mode}")

        if self.connected:
            try:
                import can
                with self._lock:
                    msg = can.Message(
                        arbitration_id=self.ID_CONTROL,
                        data=data,
                        is_extended_id=False
                    )
                    self.bus.send(msg)
                return True
            except Exception as e:
                print(f"  ❌ CAN 전송 오류: {e}")
                return False
        else:
            # 시뮬레이션 출력
            mode_str = {0:"STOP", 1:"AUTO", 2:"MANUAL"}.get(mode, "?")
            print(f"  [SIM-CAN] steer={steering:+.3f} "
                  f"speed={speed:.3f} mode={mode_str}")
            return True

    def send_stop(self):
        """긴급 정지"""
        self.send_control(0.0, 0.0, mode=0)

    # ── 수신 루프 (백그라운드) ────────────────
    def _rx_loop(self):
        import can
        while True:
            try:
                msg = self.bus.recv(timeout=1.0)
                if msg and msg.arbitration_id == self.ID_STATUS:
                    ready, err = struct.unpack('BB', msg.data[:2])
                    if self.verbose:
                        print(f"  [CAN RX] ready={ready} err={err}")
            except Exception:
                time.sleep(0.1)

    # ── 상태 정보 ─────────────────────────────
    def get_last_command(self) -> tuple:
        """마지막으로 전송한 (steering, speed, mode)"""
        return self._last_cmd

    def close(self):
        if self.connected:
            self.send_stop()
            time.sleep(0.1)
            try:
                self.bus.shutdown()
            except Exception:
                pass
        print("  CAN 종료")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── 간단한 연결 테스트 ────────────────────────
if __name__ == "__main__":
    print("[CAN 인터페이스 테스트]")
    print("python-can 라이브러리: pip install python-can")
    print()

    with STM32CANInterface(channel='COM3', verbose=True) as can_if:
        print("\n테스트 명령 전송 (5회):")
        for i in range(5):
            steer = math.sin(i * 0.5) * 0.3
            speed = 0.4
            can_if.send_control(steer, speed, mode=1)
            time.sleep(0.5)
        can_if.send_stop()
        print("테스트 완료")

    import math
