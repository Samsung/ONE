from onelib import argumentparse
from onelib.argumentparse import DriverName, NormalOption, TargetOption


def command_schema():
    parser = argumentparse.ArgumentParser()
    parser.add_argument("dummy-compile", action=DriverName)
    parser.add_argument("--target", action=TargetOption)
    # multiple option names
    parser.add_argument("-o", "--output_path", action=NormalOption)
    parser.add_argument("input", action=NormalOption)

    return parser
