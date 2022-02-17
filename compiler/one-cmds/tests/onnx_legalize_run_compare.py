import onnxruntime as rt
import onnx
import sys
import os
import numpy as np

def _generate_inputs(model):
    inputs = {}
    for input in model.graph.input:
        # check if elem type is float32
        assert(input.type.tensor_type.elem_type == 1)
        input_shape = []
        for dim in input.type.tensor_type.shape.dim:
            input_shape += [dim.dim_value]
        inputs[input.name] = np.random.random(input_shape).astype(np.float32)
    return inputs

def _run_model(model, inputs):
    output_names = list(map(lambda output: output.name, model.graph.output))
    session = rt.InferenceSession(model.SerializeToString())
    outputs = session.run(output_names, inputs)
    return outputs

def _compare_resuts(ref_outputs, test_outputs, tolerance):
    num_outputs = len(ref_outputs)
    assert(len(test_outputs) == num_outputs)
    for i in range(num_outputs):
        if ref_outputs[i].shape != test_outputs[i].shape:
            print("output {} shape mismatch".format(i))
            return False
        peak_error = np.abs(ref_outputs[i] - test_outputs[i]).max()/np.abs(ref_outputs[i]).max()
        if peak_error > tolerance:
            print("output {} peak error to value ratio {} is too big".format(i, peak_error))
            return False
    return True

if __name__ == '__main__':
    # this manipulation is needed to add search path where onnx_legalizer is installed
    sys.path += os.environ['PATH'].split(':')
    import onnx_legalizer

    if len(sys.argv) < 5:
        exit('expecting 4 arguments: path to input model, path to "legalized" model,'
             'base name for generated test inputs, output tolerance')
    input_model_path = sys.argv[1]
    output_model_path = sys.argv[2]
    input_dump_path = sys.argv[3]
    tolerance = float(sys.argv[4])

    model = onnx.load(input_model_path)

    inputs = _generate_inputs(model)

    for i in inputs:
        np.save('{}_{}.npy'.format(input_dump_path, i), inputs[i])

    ref_outputs = _run_model(model, inputs)

    options = onnx_legalizer.LegalizeOptions()
    options.unroll_rnn = True
    options.unroll_lstm = True
    onnx_legalizer.legalize(model, options)

    with open(output_model_path, 'wb') as f:
        f.write(model.SerializeToString())

    test_outputs = _run_model(model, inputs)

    if not _compare_resuts(ref_outputs, test_outputs, tolerance):
        exit('comparison failed')
