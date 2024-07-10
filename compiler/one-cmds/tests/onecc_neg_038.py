from onelib import argumentparse
from onelib.argumentparse import DriverName, NormalOption, TargetOption


def command_schema():
    parser = argumentparse.ArgumentParser()
    parser.add_argument("dummy-compiler", action=DriverName)
    parser.add_argument("--target", action=TargetOption)
    # Invalid opiton: positional and optional options are mixed
    parser.add_argument("verbose", "-v", action=NormalOption)
    parser.add_argument("input", action=NormalOption)
    parser.add_argument("output", action=NormalOption)

    return parser
