import time
from dual_arm.motors import Motor, MotorNormMode
from dual_arm.motors.dynamixel import DynamixelMotorsBus, OperatingMode

def verify_motors(port):
    print(f"\nVerifying and Setting Extended Position on {port}")
    # Scan a wide range of IDs
    motors = {str(i): Motor(i, "xl430-w250", MotorNormMode.RANGE_M100_100) for i in range(1, 30)}
    bus = DynamixelMotorsBus(port=port, motors=motors)
    
    try:
        bus.connect(handshake=False)
        
        found_motors = []
        for i in range(1, 30):
            if bus.ping(i) is not None:
                found_motors.append(i)
        
        if not found_motors:
            print("No motors found.")
            return

        print(f"Found motors: {found_motors}")
        
        for m_id in found_motors:
            # 1. Disable torque
            bus.write("Torque_Enable", str(m_id), 0)
            time.sleep(0.05)
            
            # 2. Set to EXTENDED_POSITION (4)
            print(f"ID {m_id}: Setting Operating_Mode to 4...")
            bus.write("Operating_Mode", str(m_id), 4)
            time.sleep(0.1)
            
            # 3. Verify
            mode = bus.read("Operating_Mode", str(m_id), normalize=False)
            print(f"ID {m_id}: Current Operating_Mode = {mode}")
            
            # 4. Read Position
            pos = bus.read("Present_Position", str(m_id), normalize=False)
            print(f"ID {m_id}: Present_Position = {pos}")
            
            # 5. Enable torque (Optional, for leader we usually keep it disabled, for follower enabled)
            # bus.write("Torque_Enable", str(m_id), 1)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    verify_motors("/dev/ttyUSB0")
