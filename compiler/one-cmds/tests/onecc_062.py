from onelib import argparse
from onelib.argparse import DriverName, NormalOption, TargetOption


def command_schema():
    parser = argparse.ArgumentParser()
    parser.add_argument("dummy-compile", action=DriverName)
    parser.add_argument("--target", action=TargetOption)
    parser.add_argument("--DSP-quota", action=NormalOption)
    parser.add_argument("-o", action=NormalOption)
    parser.add_argument("input", action=NormalOption)

    return parser
