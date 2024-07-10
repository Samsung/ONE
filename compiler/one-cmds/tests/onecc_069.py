from onelib import argumentparse
from onelib.argumentparse import DriverName, NormalOption, TargetOption


def command_schema():
    parser = argumentparse.ArgumentParser()
    parser.add_argument("dummy-compiler", action=DriverName)
    parser.add_argument("--target", action=TargetOption)
    # multiple options
    parser.add_argument("--verbose", "-v", action=NormalOption, dtype=bool)
    parser.add_argument("input", action=NormalOption)
    parser.add_argument("output", action=NormalOption)

    return parser
