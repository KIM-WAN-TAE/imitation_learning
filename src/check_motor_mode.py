import logging
from dual_arm.motors import Motor, MotorNormMode
from dual_arm.motors.dynamixel import DynamixelMotorsBus, OperatingMode

logging.basicConfig(level=logging.INFO)

def check_port(port):
    print(f"\nChecking port: {port}")
    baudrates = [1000000, 57600, 115200, 2000000, 3000000]
    
    for baud in baudrates:
        print(f"  Trying baudrate: {baud}")
        # Dummy motors dict for initialization
        motors = {str(i): Motor(i, "xl430-w250", MotorNormMode.RANGE_M100_100) for i in range(1, 11)}
        
        bus = DynamixelMotorsBus(port=port, motors=motors)
        bus.default_baudrate = baud
        try:
            bus.connect(handshake=False)
            bus.set_baudrate(baud)
            
            found_motors = []
            for i in range(1, 15): # Scan a bit more IDs
                model = bus.ping(i)
                if model is not None:
                    found_motors.append(i)
            
            if not found_motors:
                bus.disconnect(disable_torque=False)
                continue

            print(f"  Found motors with IDs: {found_motors} at {baud} bps")
            
            for motor_id in found_motors:
                try:
                    mode = bus.read("Operating_Mode", str(motor_id), normalize=False)
                    mode_name = "Unknown"
                    try:
                        mode_name = OperatingMode(mode).name
                    except ValueError:
                        pass
                    print(f"  ID {motor_id}: Mode = {mode} ({mode_name})")
                except Exception as e:
                    print(f"  ID {motor_id}: Failed to read mode: {e}")
            
            bus.disconnect(disable_torque=False)
            return # Found motors on this port, no need to try other baudrates
                    
        except Exception as e:
            # print(f"    Error on {port} at {baud}: {e}")
            pass
        finally:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    check_port("/dev/ttyUSB0")
    check_port("/dev/ttyUSB1")
