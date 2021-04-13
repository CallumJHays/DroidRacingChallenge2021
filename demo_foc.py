from serial import Serial
import pyvesc
from pyvesc import GetRotorPosition, GetValues, SetCurrent, SetCurrentBrake, SetDutyCycle, SetPosition, SetRotorPositionMode, SetRPM
from typing import Any


class MiniFocPlus(Serial):
    # https://github.com/UofSSpaceTeam/roveberrypy/blob/a0bf9ada77ad0daf93c25ff917d3043f9da4cb66/roverprocess/USBServer.py

    def __init__(self, port: str = '/dev/ttyACM0'):
        super().__init__(port, 115200)
        self.buf = b''

    def command(self, cmd: Any):
        if cmd is pyvesc.GetRotorPosition or pyvesc.GetValues:
            fn = pyvesc.encode_request
        else:
            fn = pyvesc.encode
        self.write(fn(cmd))
        while True:
            more = self.read_all()
            if more:
                self.buf += more
                res, consumed = pyvesc.decode(self.buf)
                if res:
                    self.buf = self.buf[consumed:]
                    return res


with MiniFocPlus() as foc:
    cmd = SetRPM()
    vals = foc.command(SetRPM())
    # pos = foc.get_rotor_pos()
    print(pos)
