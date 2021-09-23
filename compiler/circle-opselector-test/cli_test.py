import os

COLOR = {
    "BLUE": "\033[94m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}
# set path
SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_PATH = '/'.join(SCRIPT_PATH.split('/')[:-1])
os.chdir(SCRIPT_PATH + '/../../')
ROOT_PATH = os.getcwd()
OPSELECTOR_PATH = f'{ROOT_PATH}/build/compiler/circle-opselector'
TFLITE_PATH = f'{ROOT_PATH}/build/compiler/common-artifacts'


def run_opselector(option, tflite_name):
    """
    Run CircleOpselector
    """
    result = os.system(f'{OPSELECTOR_PATH}/opselector {option} --input {TFLITE_PATH}/{tflite_name} --output {TFLITE_PATH}/{tflite_name} > /dev/null')
    return result


tflite_name = 'Part_Sqrt_Rsqrt_002.circle'  # the number of operators is 4
options = {
    # by_id
    '--by_id "1,2"': 0,  
    '--by_id "1, 2"': 0, 
    '--by_id "1-2"': 0,  
    '--by_id "3, 1"': 0,
    '--by_id "1 - 2"': 0, 
    '--by_id "0, 0, 1"': 0,  # duplicaged nodes -> 0, 1

    '--by_id "1,4"': 1,  # out of range operator
    '--by_id "0-5"': 1,    # out of range operator
    '--by_id "a,b"': 1,  # not integer 
    '--by_id "0.1"': 1,  # not integer 
    '--by_id "0--1"': 1,  # double hyphen
    '--by_id "0.1"': 1,
    # by_name
}

for option, value in options.items():
    result = run_opselector(option, tflite_name) 
    if result == value:
        print(COLOR['BLUE'], 'pass', option, f'expected: {value}, result: {result}', COLOR['ENDC'])
    else:
        print(COLOR['RED'], 'fail', option, f'expected: {value}, result: {result}', COLOR['ENDC'])
