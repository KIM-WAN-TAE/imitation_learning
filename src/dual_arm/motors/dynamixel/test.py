import dynamixel_sdk as dxl

PORT="/dev/ttyUSB0"

def try_baud(baud):
    print("\n=== baud:", baud, "===")
    ph = dxl.PortHandler(PORT)
    if not ph.openPort():
        print("openPort failed")
        return
    if not ph.setBaudRate(baud):
        print("setBaudRate failed")
        ph.closePort()
        return

    # Protocol 2.0 broadcast ping
    p2 = dxl.PacketHandler(2.0)
    try:
        data_list, comm = p2.broadcastPing(ph)
        if comm == dxl.COMM_SUCCESS:
            found = {i: data[0] for i, data in data_list.items()}
            print("P2 broadcast found:", found)
        else:
            print("P2 broadcast comm err:", p2.getTxRxResult(comm))
    except Exception as e:
        print("P2 broadcast exception:", e)

    # Protocol 1.0 ping some ids (0~30만 우선)
    p1 = dxl.PacketHandler(1.0)
    found1 = {}
    for mid in range(0, 31):
        model, comm, err = p1.ping(ph, mid)
        if comm == dxl.COMM_SUCCESS and err == 0:
            found1[mid] = int(model)
    print("P1 ping found:", found1)

    ph.closePort()

# 자주 쓰는 baud만 먼저 (필요하면 더 늘리세요)
for b in [1000000, 57600, 115200, 2000000, 3000000]:
    try_baud(b)