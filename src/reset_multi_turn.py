import time
from dual_arm.motors import Motor, MotorNormMode
from dual_arm.motors.dynamixel import DynamixelMotorsBus, OperatingMode

def reset_motors(port):
    print(f"\nResetting motors on {port}")
    motors = {str(i): Motor(i, "xl430-w250", MotorNormMode.RANGE_M100_100) for i in range(1, 15)}
    bus = DynamixelMotorsBus(port=port, motors=motors)
    
    try:
        bus.connect(handshake=False)
        
        found_motors = []
        for i in range(1, 15):
            if bus.ping(i) is not None:
                found_motors.append(i)
        
        if not found_motors:
            print("No motors found.")
            return

        print(f"Found motors: {found_motors}")
        
        # Disable torque first
        for m_id in found_motors:
            bus.write("Torque_Enable", str(m_id), 0)
        
        for m_id in found_motors:
            print(f"Configuring ID {m_id} to EXTENDED_POSITION...")
            # First set to POSITION to reset multi-turn if needed
            bus.write("Operating_Mode", str(m_id), OperatingMode.POSITION.value)
            time.sleep(0.1)
            # Then set to EXTENDED_POSITION
            bus.write("Operating_Mode", str(m_id), OperatingMode.EXTENDED_POSITION.value)
            time.sleep(0.1)
            
            # Reset Homing Offset to 0 to avoid confusion
            bus.write("Homing_Offset", str(m_id), 0)
            
            # Enable torque
            bus.write("Torque_Enable", str(m_id), 1)
            
            # Read current position
            pos = bus.read("Present_Position", str(m_id), normalize=False)
            print(f"ID {m_id} initialized. Current Position: {pos}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    reset_motors("/dev/ttyUSB0")
