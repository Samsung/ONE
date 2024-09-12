from onelib import argumentparse
from onelib.argumentparse import DriverName, NormalOption, TargetOption


def command_schema():
    parser = argumentparse.ArgumentParser()
    parser.add_argument("dummyV2-compiler", action=DriverName)
    parser.add_argument("--target", action=TargetOption)
    parser.add_argument("--command", action=NormalOption, dtype=bool)

    return parser
