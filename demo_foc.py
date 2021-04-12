from serial import Serial
import pyvesc

# https://github.com/UofSSpaceTeam/roveberrypy/blob/a0bf9ada77ad0daf93c25ff917d3043f9da4cb66/roverprocess/USBServer.py
with Serial('/dev/ttyACM0', 115200) as ser:
    req = pyvesc.GetValues
    ser.write(pyvesc.encode_request(req))
    in_buf = b''
    while ser.in_waiting > 0:
        in_buf += ser.read(ser.in_waiting)
    res, consumed = pyvesc.decode(in_buf)
    print()
    print(res)
