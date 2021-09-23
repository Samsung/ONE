import os
import flatbuffers
import tensorflow as tf

script_path = os.path.realpath(__file__)
script_path = '/'.join(script_path.split('/')[:-1])
os.chdir(script_path)

with open('test.lst') as f:
    tflite_files = f.readlines()
TFLITE_FILES_FOLDER = '../../build/compiler/common-artifacts/'
tflite_files = [x.strip()+'.tflite' for x in tflite_files]


def get_operator_num(tflite_file):
    interpreter = tf.lite.Interpreter(TFLITE_FILES_FOLDER + tflite_file)
    interpreter.allocate_tensors()
    tensor_count = len(interpreter.get_tensor_details())
    input_tensor_count = len(interpreter.get_input_details())
    output_tensor_count = len(interpreter.get_output_details())
    op_num = tensor_count - input_tensor_count
    return op_num

# Check case 1 (continuous select)
def gen_continuos_cases(op_num):
    """
    if op_num : 4, return generator [0, 1], [0, 1, 2], [0, 1, 2, 3], [1, 2], ... [2, 3]
    """
    for i in range(op_num):
        for j in range(i+1, op_num):
            select_nodes = list(range(i, j+1))
            select_nodes_str = ' '.join([str(x) for x in select_nodes])
            select_nodes_str = '\"' + select_nodes_str + '\"'
            yield select_nodes_str


op_num = get_operator_num(tflite_files[0])
for tflite_file in tflite_files:
    op_num = get_operator_num(tflite_file)
    op_gen = gen_continuos_cases(op_num)
    for select_nodes in op_gen:
        test_result = os.system(f'./opselector_test.sh {tflite_file} {select_nodes}')
