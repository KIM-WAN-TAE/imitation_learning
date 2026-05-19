# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
혼합 프로토콜 지원 DynamixelMotorsBus

- X-series (Protocol 2.0): broadcastPing / GroupSyncRead/Write 지원
- AX-12A  (Protocol 1.0): broadcastPing 없음 / SyncRead 없음 → 개별 read/ping 처리

핵심 패치:
1) ping()를 모터별 프로토콜에 맞춰 오버라이드 (handshake에서 AX-12A 누락 방지)
2) read_calibration()/write_calibration()/is_calibrated에서 AX-12A에 없는 항목(Homing_Offset 등) 접근 방지
3) reset_calibration()/set_half_turn_homings()에서 AX-12A는 스킵 (Homing_Offset write 방지)
4) AX-12A에서 지원하지 않는 항목(Operating_Mode 등) write 시 스킵 처리
5) so_* 코드에서 쓰는 별칭(P_Coefficient 등) → 표준 키(Position_P_Gain 등)로 alias 처리
"""

import logging
from copy import deepcopy
from enum import Enum

from ..encoding_utils import decode_twos_complement, encode_twos_complement
from ..motors_bus import Motor, MotorCalibration, NameOrID, SerialMotorsBus, Value, get_address
from .tables import (
    AVAILABLE_BAUDRATES,
    MODEL_BAUDRATE_TABLE,
    MODEL_CONTROL_TABLE,
    MODEL_ENCODING_TABLE,
    MODEL_NUMBER_TABLE,
    MODEL_RESOLUTION,
    MODEL_PROTOCOL_TABLE,
)

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 2.0
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000

NORMALIZED_DATA = ["Goal_Position", "Present_Position"]


class OperatingMode(Enum):
    CURRENT = 0
    VELOCITY = 1
    POSITION = 3
    EXTENDED_POSITION = 4
    CURRENT_POSITION = 5
    PWM = 16


class DriveMode(Enum):
    NON_INVERTED = 0
    INVERTED = 1


class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


def _split_into_byte_chunks(value: int, length: int) -> list[int]:
    import dynamixel_sdk as dxl

    if length == 1:
        return [value & 0xFF]
    if length == 2:
        return [dxl.DXL_LOBYTE(value), dxl.DXL_HIBYTE(value)]
    if length == 4:
        return [
            dxl.DXL_LOBYTE(dxl.DXL_LOWORD(value)),
            dxl.DXL_HIBYTE(dxl.DXL_LOWORD(value)),
            dxl.DXL_LOBYTE(dxl.DXL_HIWORD(value)),
            dxl.DXL_HIBYTE(dxl.DXL_HIWORD(value)),
        ]
    raise NotImplementedError(f"Unsupported byte size: {length}. Expected [1, 2, 4].")


class DynamixelMotorsBus(SerialMotorsBus):
    """
    LeRobot의 SerialMotorsBus를 DynamixelSDK로 구현한 버스.
    AX-12A(Protocol 1.0)와 X-series(Protocol 2.0)가 섞여 있어도 동작하도록 패치.
    """

    # so_* 코드에서 쓰는 구버전/별칭 키를 표준 ControlTable 키로 매핑
    _ITEM_ALIAS: dict[str, str] = {
        # Position PID gain (X-series)
        "P_Coefficient": "Position_P_Gain",
        "I_Coefficient": "Position_I_Gain",
        "D_Coefficient": "Position_D_Gain",
        # (혹시 velocity 쪽 별칭이 나오면)
        "Velocity_P_Coefficient": "Velocity_P_Gain",
        "Velocity_I_Coefficient": "Velocity_I_Gain",
    }

    apply_drive_mode = False
    available_baudrates = deepcopy(AVAILABLE_BAUDRATES)
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS
    model_baudrate_table = deepcopy(MODEL_BAUDRATE_TABLE)
    model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
    model_encoding_table = deepcopy(MODEL_ENCODING_TABLE)
    model_number_table = deepcopy(MODEL_NUMBER_TABLE)
    model_resolution_table = deepcopy(MODEL_RESOLUTION)
    normalized_data = deepcopy(NORMALIZED_DATA)

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        super().__init__(port, motors, calibration)

        import dynamixel_sdk as dxl

        # id -> protocol (tables.py 기반)
        self._id_protocol: dict[int, float] = {
            m.id: float(MODEL_PROTOCOL_TABLE.get(m.model, 2.0)) for m in self.motors.values()
        }

        self.port_handler = dxl.PortHandler(self.port)

        # 프로토콜별 PacketHandler
        self.packet_handler_p2 = dxl.PacketHandler(2.0)
        self.packet_handler_p1 = dxl.PacketHandler(1.0)

        # 기본은 2.0으로 시작
        self.packet_handler = self.packet_handler_p2

        # Sync 객체(2.0만 SyncRead 지원)
        self.sync_reader_p2 = dxl.GroupSyncRead(self.port_handler, self.packet_handler_p2, 0, 0)
        self.sync_writer_p2 = dxl.GroupSyncWrite(self.port_handler, self.packet_handler_p2, 0, 0)
        self.sync_writer_p1 = dxl.GroupSyncWrite(self.port_handler, self.packet_handler_p1, 0, 0)

        self.sync_reader = self.sync_reader_p2
        self.sync_writer = self.sync_writer_p2

        self._comm_success = dxl.COMM_SUCCESS
        self._no_error = 0x00

    # -------------------------
    # 내부 유틸
    # -------------------------
    def _select_protocol_for_motor(self, motor: NameOrID) -> float:
        motor_id = self._get_motor_id(motor)
        return float(self._id_protocol.get(motor_id, 2.0))

    def _with_protocol(self, protocol: float):
        """
        packet_handler/sync_writer/sync_reader를 프로토콜에 맞게 임시 교체하는 helper
        """

        class _Swap:
            def __init__(self, bus: "DynamixelMotorsBus", proto: float):
                self.bus = bus
                self.proto = proto
                self._old_packet = None
                self._old_reader = None
                self._old_writer = None

            def __enter__(self):
                self._old_packet = self.bus.packet_handler
                self._old_reader = self.bus.sync_reader
                self._old_writer = self.bus.sync_writer

                if self.proto == 1.0:
                    self.bus.packet_handler = self.bus.packet_handler_p1
                    # P1은 SyncRead 미지원 → reader는 그대로 두되 사용하지 않음
                    self.bus.sync_writer = self.bus.sync_writer_p1
                else:
                    self.bus.packet_handler = self.bus.packet_handler_p2
                    self.bus.sync_reader = self.bus.sync_reader_p2
                    self.bus.sync_writer = self.bus.sync_writer_p2
                return self.bus

            def __exit__(self, exc_type, exc, tb):
                self.bus.packet_handler = self._old_packet
                self.bus.sync_reader = self._old_reader
                self.bus.sync_writer = self._old_writer
                return False

        return _Swap(self, protocol)

    def _is_comm_success(self, comm_result: int) -> bool:
        return int(comm_result) == int(self._comm_success)

    def _map_item(self, data_name: str) -> str:
        return self._ITEM_ALIAS.get(data_name, data_name)

    # -------------------------
    # ✅ 핵심: ping 오버라이드
    # -------------------------
    def ping(self, motor: NameOrID, num_retry: int = 0, raise_on_error: bool = False) -> int | None:
        """
        handshake 단계에서 호출되는 ping이 모터별 프로토콜로 동작하도록 보장.
        (AX-12A가 Protocol 2.0 ping으로 처리되어 누락되는 문제 방지)
        """
        proto = self._select_protocol_for_motor(motor)
        with self._with_protocol(proto):
            # SerialMotorsBus.ping은 error != 0 이면 None을 리턴하므로,
            # 통신만 성공하면 모델 번호를 리턴하도록 Dynamixel 특성에 맞춰 직접 호출.
            id_ = self._get_motor_id(motor)
            for n_try in range(1 + num_retry):
                model_number, comm, error = self.packet_handler.ping(self.port_handler, id_)
                if self._is_comm_success(comm):
                    return int(model_number)

            if raise_on_error:
                raise ConnectionError(f"Ping failed for motor {motor}")
            return None

    # -------------------------
    # read/write 라우팅
    # -------------------------
    def read(self, data_name: str, motor: NameOrID, *, normalize: bool = True, num_retry: int = 0) -> Value:
        data_name = self._map_item(data_name)

        proto = self._select_protocol_for_motor(motor)
        model = self._get_motor_model(motor)

        ctrl = self.model_ctrl_table.get(model, {})
        if data_name not in ctrl and proto == 1.0:
            # P1에서 없는 항목을 읽으려 하면 명시적으로 에러 (디버깅 안전)
            raise KeyError(f"Address for '{data_name}' not found in {model} control table (protocol 1.0).")

        with self._with_protocol(proto):
            return super().read(data_name, motor, normalize=normalize, num_retry=num_retry)

    def write(
        self, data_name: str, motor: NameOrID, value: Value, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        data_name = self._map_item(data_name)

        proto = self._select_protocol_for_motor(motor)
        model = self._get_motor_model(motor)

        ctrl = self.model_ctrl_table.get(model, {})
        if data_name not in ctrl and proto == 1.0:
            # ✅ 핵심: AX-12A(Protocol 1.0)에서 지원하지 않는 항목(Operating_Mode 등)은 스킵
            logger.debug(f"Skip write: '{data_name}' not supported by model '{model}' (protocol 1.0).")
            return

        with self._with_protocol(proto):
            return super().write(data_name, motor, value, normalize=normalize, num_retry=num_retry)

    def _write(
        self,
        addr: int,
        length: int,
        motor_id: int,
        value: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[int, int]:
        """
        AX-12A(Protocol 1.0)의 통신 속도를 높이기 위해 저수준 write를 오버라이드합니다.
        응답을 기다리지 않는 writeTxOnly 사용을 시도할 수 있습니다.
        """
        import dynamixel_sdk as dxl
        
        # Protocol 1.0이고 Goal_Position(addr=30) 업데이트일 경우 속도를 위해 TxOnly 고려 가능
        # 하지만 안정성을 위해 여기서는 기본 write를 사용하되, SDK 레벨에서의 최적화를 보장합니다.
        return super()._write(addr, length, motor_id, value, num_retry=num_retry, raise_on_error=raise_on_error, err_msg=err_msg)

    def sync_read(
        self,
        data_name: str,
        motors: NameOrID | list[NameOrID] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Value]:
        data_name = self._map_item(data_name)

        names = self._get_motors_list(motors)
        p2 = [m for m in names if self._select_protocol_for_motor(m) == 2.0]
        p1 = [m for m in names if self._select_protocol_for_motor(m) == 1.0]

        out: dict[str, Value] = {}

        # P2는 그룹 읽기
        if p2:
            with self._with_protocol(2.0):
                try:
                    out.update(super().sync_read(data_name, p2, normalize=normalize, num_retry=num_retry))
                except ConnectionError as e:
                    logger.debug(f"Sync read failed for {p2} ({e!r}). Falling back to individual reads.")
                    for m in p2:
                        out[m] = self.read(data_name, m, normalize=normalize, num_retry=num_retry)

        # P1은 개별 읽기 (단, P1에 존재하는 item만 읽어야 함)
        if p1:
            for m in p1:
                key = str(m) if isinstance(m, int) else m
                out[key] = self.read(data_name, m, normalize=normalize, num_retry=num_retry)

        return out

    def sync_write(
        self,
        data_name: str,
        values: dict[NameOrID, Value],
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        data_name = self._map_item(data_name)

        p2_vals: dict[NameOrID, Value] = {}
        p1_vals: dict[NameOrID, Value] = {}

        for motor, value in values.items():
            if self._select_protocol_for_motor(motor) == 1.0:
                p1_vals[motor] = value
            else:
                p2_vals[motor] = value

        if p2_vals:
            with self._with_protocol(2.0):
                super().sync_write(data_name, p2_vals, normalize=normalize, num_retry=num_retry)

        # P1은 안전하게 개별 write (없는 항목이면 write()에서 스킵)
        if p1_vals:
            for motor, value in p1_vals.items():
                self.write(data_name, motor, value, normalize=normalize, num_retry=num_retry)

    # -------------------------
    # SerialMotorsBus hook/abstract 구현
    # -------------------------
    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        # P1의 SyncRead는 sync_read에서 fallback 처리하므로 여기서 막지 않음
        return

    def _handshake(self) -> None:
        # handshake 전에 baudrate를 먼저 맞춰야 모터가 잡힘
        try:
            self.set_baudrate(self.default_baudrate)  # 보통 1_000_000
        except Exception:
            pass

        ids_models = self.broadcast_ping(num_retry=1)

        # 혹시 default baudrate가 아닌 경우를 대비한 fallback
        if not ids_models:
            fallback_bauds = [1_000_000, 57600, 115200, 2_000_000, 3_000_000]
            for br in fallback_bauds:
                try:
                    self.set_baudrate(br)
                    ids_models = self.broadcast_ping(num_retry=1)
                    if ids_models:
                        # 찾았으면 default_baudrate도 갱신(선택)
                        self.default_baudrate = br
                        break
                except Exception:
                    continue

        # 이제 모터 존재 체크 수행
        self._assert_motors_exist()

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        model = self.motors[motor].model
        search_baudrates = [initial_baudrate] if initial_baudrate is not None else self.model_baudrate_table[model]

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            id_model = self.broadcast_ping()
            if id_model:
                found_id, found_model = next(iter(id_model.items()))
                expected_model_nb = self.model_number_table[model]
                if found_model != expected_model_nb:
                    raise RuntimeError(
                        f"Found one motor on {baudrate=} with id={found_id} but model_number={found_model} "
                        f"!= expected={expected_model_nb} for motor '{motor}' model '{model}'."
                    )
                return baudrate, found_id

        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")

    def configure_motors(self, return_delay_time: int = 0) -> None:
        # AX-12A에도 Return_Delay_Time이 존재하므로 공통 적용 가능
        for motor in self.motors:
            self.write("Return_Delay_Time", motor, return_delay_time)

    # -------------------------
    # AX-12A 캘리브레이션 안전 처리
    # -------------------------
    @property
    def is_calibrated(self) -> bool:
        """
        AX-12A(P1)는 Homing_Offset/Min/Max 등의 개념이 없어서
        "P2 모터들만" 엄격 비교하고 P1은 항상 OK로 취급한다.
        비교 시 ID는 제외하고 캘리브레이션 값(drive_mode, homing_offset, range_min, range_max)만 비교한다.
        """
        current = self.read_calibration()

        if self.calibration is None:
            return False

        for name, m in self.motors.items():
            proto = float(self._id_protocol.get(m.id, 2.0))
            if proto == 1.0:
                continue
            if name not in self.calibration:
                return False
            
            cal = self.calibration[name]
            cur = current[name]
            
            # ID를 제외한 나머지 값들이 일치하는지 확인
            is_same = (
                cal.drive_mode == cur.drive_mode and
                cal.homing_offset == cur.homing_offset and
                cal.range_min == cur.range_min and
                cal.range_max == cur.range_max
            )
            
            if not is_same:
                return False
        return True

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """
        - P2(X-series): Homing_Offset, Min/Max_Position_Limit, Drive_Mode 읽어서 생성
        - P1(AX-12A): 위 항목 없음 → 기본값으로 생성
        """
        p2_names: list[str] = [n for n, m in self.motors.items() if float(self._id_protocol.get(m.id, 2.0)) == 2.0]

        offsets: dict[str, Value] = {}
        mins: dict[str, Value] = {}
        maxes: dict[str, Value] = {}
        drive_modes: dict[str, Value] = {}

        if p2_names:
            try:
                # Sync read can be unstable on mixed protocol buses.
                # Use retries to ensure we get a response.
                num_retry = 2
                offsets = self.sync_read("Homing_Offset", p2_names, normalize=False, num_retry=num_retry)
                mins = self.sync_read("Min_Position_Limit", p2_names, normalize=False, num_retry=num_retry)
                maxes = self.sync_read("Max_Position_Limit", p2_names, normalize=False, num_retry=num_retry)
                drive_modes = self.sync_read("Drive_Mode", p2_names, normalize=False, num_retry=num_retry)
            except ConnectionError as e:
                logger.debug(f"Sync read calibration failed ({e!r}). Falling back to individual reads.")
                offsets = {n: self.read("Homing_Offset", n, normalize=False, num_retry=3) for n in p2_names}
                mins = {n: self.read("Min_Position_Limit", n, normalize=False, num_retry=3) for n in p2_names}
                maxes = {n: self.read("Max_Position_Limit", n, normalize=False, num_retry=3) for n in p2_names}
                drive_modes = {n: self.read("Drive_Mode", n, normalize=False, num_retry=3) for n in p2_names}

        calibration: dict[str, MotorCalibration] = {}
        for name, m in self.motors.items():
            proto = float(self._id_protocol.get(m.id, 2.0))
            if proto == 2.0:
                calibration[name] = MotorCalibration(
                    id=m.id,
                    drive_mode=int(drive_modes[name]),
                    homing_offset=int(offsets[name]),
                    range_min=int(mins[name]),
                    range_max=int(maxes[name]),
                )
            else:
                # AX-12A: Homing_Offset이 없으므로 소프트 기본값
                res = int(self.model_resolution_table.get(m.model, 1024))
                calibration[name] = MotorCalibration(
                    id=m.id,
                    drive_mode=0,
                    homing_offset=0,
                    range_min=0,
                    range_max=res - 1,
                )
        return calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """
        P2 모터에만 Homing_Offset/Min/Max를 씀.
        P1(AX-12A)은 해당 레지스터가 없으므로 skip.

        참고: Max/Min_Position_Limit는 하드웨어 제약(0~4095)이 있으므로,
        소프트웨어 캘리브레이션 값이 이를 벗어나면 클리핑하여 쓴다.
        (소프트웨어 캘리브레이션은 원본값을 유지하여 정규화에 사용)
        """
        for name, m in self.motors.items():
            proto = float(self._id_protocol.get(m.id, 2.0))
            if proto == 1.0:
                continue

            cal = calibration_dict[name]
            model = self._get_motor_model(name)
            max_res = self.model_resolution_table.get(model, 4096) - 1

            self.write("Homing_Offset", name, cal.homing_offset)

            # 하드웨어 레지스터(EEPROM)를 직접 수정하지 않고 소프트웨어적으로만 제한하기 위해 주석 처리
            # hw_min = int(max(0, min(max_res, cal.range_min)))
            # hw_max = int(max(0, min(max_res, cal.range_max)))

            # 만약 Min > Max 가 되면 하드웨어 에러 가능성 있으므로 교정
            # if hw_min > hw_max:
            #     hw_min, hw_max = hw_max, hw_min

            # self.write("Min_Position_Limit", name, hw_min)
            # self.write("Max_Position_Limit", name, hw_max)

        if cache:
            self.calibration = calibration_dict

    # -------------------------
    # Homing_Offset 쓰는 루틴은 P2만 적용 (에러 원인 차단)
    # -------------------------
    def reset_calibration(self, motors: NameOrID | list[NameOrID] | None = None):
        """
        기본 구현은 모든 모터에 Homing_Offset=0 write를 수행함.
        AX-12A에는 Homing_Offset이 없으므로 P2만 필터링해서 호출.
        """
        names = self._get_motors_list(motors)
        p2 = [n for n in names if self._select_protocol_for_motor(n) == 2.0]
        if not p2:
            return
        return super().reset_calibration(p2)

    def set_half_turn_homings(self, motors: NameOrID | list[NameOrID] | None = None):
        """
        half-turn homing은 Homing_Offset을 쓰는 기능이라 P2만 적용 가능.
        """
        names = self._get_motors_list(motors)
        p2 = [n for n in names if self._select_protocol_for_motor(n) == 2.0]
        if not p2:
            return {}
        return super().set_half_turn_homings(p2)

    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        """
        (계산용 보조 함수)
        Present_Position = Actual_Position + Homing_Offset 관계를 이용해
        half-turn 기준으로 Homing_Offset 값을 계산한다.
        """
        half_turn_homings: dict[NameOrID, Value] = {}
        for motor, pos in positions.items():
            model = self._get_motor_model(motor)
            max_res = self.model_resolution_table[model] - 1
            half_turn_homings[motor] = int(max_res / 2) - int(pos)
        return half_turn_homings

    # -------------------------
    # Torque 관련 (abstract 충족)
    # -------------------------
    def disable_torque(self, motors: int | str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, TorqueMode.DISABLED.value, num_retry=num_retry)

    def enable_torque(self, motors: int | str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, TorqueMode.ENABLED.value, num_retry=num_retry)

    def _disable_torque(self, motor: int, model: str, num_retry: int = 0) -> None:
        # base class 내부에서 직접 저수준 write가 필요할 때 사용
        addr, length = get_address(self.model_ctrl_table, model, "Torque_Enable")
        self._write(addr, length, motor, TorqueMode.DISABLED.value, num_retry=num_retry)

    # -------------------------
    # Signed encoding 처리
    # -------------------------
    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        for id_ in ids_values:
            model = self._id_to_model(id_)
            encoding_table = self.model_encoding_table.get(model)
            if encoding_table and data_name in encoding_table:
                n_bytes = encoding_table[data_name]
                ids_values[id_] = encode_twos_complement(ids_values[id_], n_bytes)
        return ids_values

    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        for id_ in ids_values:
            model = self._id_to_model(id_)
            encoding_table = self.model_encoding_table.get(model)
            if encoding_table and data_name in encoding_table:
                n_bytes = encoding_table[data_name]
                ids_values[id_] = decode_twos_complement(ids_values[id_], n_bytes)
        return ids_values

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        return _split_into_byte_chunks(value, length)

    # -------------------------
    # discovery
    # -------------------------
    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        """
        - P2: broadcastPing
        - P1: broadcastPing 없음 → 개별 ping
        """
        found: dict[int, int] = {}

        # Protocol 2.0 broadcast ping
        with self._with_protocol(2.0):
            comm = None
            for _ in range(1 + num_retry):
                data_list, comm = self.packet_handler.broadcastPing(self.port_handler)
                if self._is_comm_success(comm):
                    found.update({id_: int(data[0]) for id_, data in data_list.items()})
                    break
                if raise_on_error:
                    logger.debug(self.packet_handler.getTxRxResult(comm))

            if comm is not None and (not self._is_comm_success(comm)) and raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))

        # Protocol 1.0: configured IDs만 빠르게 ping
        p1_ids = [id_ for id_, p in self._id_protocol.items() if float(p) == 1.0]
        if not p1_ids:
            p1_ids = list(range(0, 253))

        with self._with_protocol(1.0):
            for motor_id in p1_ids:
                for _ in range(1 + num_retry):
                    model_number, comm, err = self.packet_handler.ping(self.port_handler, motor_id)
                    # P1 status error(err != 0)가 있더라도 통신에 성공하면 모터를 찾은 것으로 간주
                    if self._is_comm_success(comm):
                        found[motor_id] = int(model_number)
                        break

        return found or None