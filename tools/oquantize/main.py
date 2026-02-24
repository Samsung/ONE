import ctypes
import os
import sys
import numpy as np
import flatbuffers

# Add tools/o2o to sys.path to import circle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../o2o')))
try:
    import circle
except ImportError:
    print(
        "Error: Could not import 'circle'. Make sure tools/o2o is in PYTHONPATH or the script is run from the correct location."
    )
    sys.exit(1)


def load_ggml_library():
    lib_path = os.path.join(os.path.dirname(__file__), 'lib', 'libggml_quant.so')
    if not os.path.exists(lib_path):
        print(f"Error: {lib_path} not found. Please build the package first.")
        sys.exit(1)

    lib = ctypes.CDLL(lib_path)

    # void quantize_row_q4_0(const float * x, void * y, int64_t k);
    lib.quantize_row_q4_0.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64
    ]
    lib.quantize_row_q4_0.restype = None

    return lib


def quantize_tensor(lib, tensor_data):
    # tensor_data is a numpy array of float32
    k = tensor_data.size

    if k % 32 != 0:
        print(f"Warning: Tensor size {k} is not a multiple of 32. Skipping quantization.")
        return None

    # QK4_0 = 32
    # block_q4_0 size = sizeof(ggml_half) + QK4_0 / 2 = 2 + 16 = 18 bytes
    block_size = 18
    num_blocks = k // 32
    output_size = num_blocks * block_size

    output_buffer = (ctypes.c_byte * output_size)()

    # Create a pointer to the input data
    input_ptr = tensor_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call the C function
    # quantize_row_q4_0 processes the whole row (k elements)
    lib.quantize_row_q4_0(input_ptr, output_buffer, ctypes.c_int64(k))

    return bytearray(output_buffer)


def main():
    if len(sys.argv) != 4:
        print("Usage: python -m oquantize <quant_type> <input_circle> <output_circle>")
        print("Supported quant_type: q4_0")
        sys.exit(1)

    quant_type = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    if quant_type != "q4_0":
        print(
            f"Error: Unsupported quantization type '{quant_type}'. Only 'q4_0' is supported."
        )
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)

    lib = load_ggml_library()

    print(f"Loading {input_path}...")
    with open(input_path, 'rb') as f:
        buf = f.read()

    model = circle.Model.GetRootAs(buf, 0)
    model_t = circle.ModelT.InitFromObj(model)

    quantized_count = 0

    for subgraph in model_t.subgraphs:
        for op in subgraph.operators:
            target_tensor_idx = -1

            if op.opcodeIndex < len(model_t.operatorCodes):
                op_code = model_t.operatorCodes[op.opcodeIndex]
                builtin_code = op_code.builtinCode

                if builtin_code == circle.BuiltinOperator.GATHER:
                    # GATHER: input 0 is params (weights)
                    if len(op.inputs) > 0:
                        target_tensor_idx = op.inputs[0]
                elif builtin_code == circle.BuiltinOperator.FULLY_CONNECTED:
                    # FULLY_CONNECTED: input 1 is weights
                    if len(op.inputs) > 1:
                        target_tensor_idx = op.inputs[1]

            if target_tensor_idx != -1:
                tensor = subgraph.tensors[target_tensor_idx]

                if tensor.type == circle.TensorType.FLOAT32:
                    buffer_idx = tensor.buffer
                    if buffer_idx < len(model_t.buffers):
                        buffer_obj = model_t.buffers[buffer_idx]

                        # Check if buffer has data
                        if buffer_obj.data is not None:
                            # Convert to numpy array
                            # buffer_obj.data is a list of ints (bytes) or numpy array
                            # circle.py generated code usually behaves like this:
                            # if InitFromObj used numpy, it might be numpy.
                            # Let's assume it's a list of uint8 or similar.

                            data_bytes = bytes(buffer_obj.data)
                            tensor_data = np.frombuffer(data_bytes, dtype=np.float32)

                            print(
                                f"Quantizing tensor {target_tensor_idx} (size={tensor_data.size})..."
                            )

                            quantized_data = quantize_tensor(lib, tensor_data)

                            if quantized_data is not None:
                                # Update buffer
                                buffer_obj.data = list(
                                    quantized_data
                                )  # FlatBuffers python expects list of ints for ubyte vector?
                                # Or numpy array? circle.py:
                                # if np is not None and type(self.data) is np.ndarray: builder.CreateNumpyVector(self.data)
                                # So we can set it to numpy array of uint8
                                buffer_obj.data = np.frombuffer(quantized_data,
                                                                dtype=np.uint8)

                                # Update tensor type
                                tensor.type = circle.TensorType.GGML_Q4_0
                                quantized_count += 1

    if quantized_count > 0:
        print(f"Quantized {quantized_count} tensors.")
        print(f"Saving to {output_path}...")

        builder = flatbuffers.Builder(1024)
        model_offset = model_t.Pack(builder)
        builder.Finish(model_offset, file_identifier=b'CIR0')

        with open(output_path, 'wb') as f:
            f.write(builder.Output())
        print("Done.")
    else:
        print("No tensors quantized.")


if __name__ == "__main__":
    main()
