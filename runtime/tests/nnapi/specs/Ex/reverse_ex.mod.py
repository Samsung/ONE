def test(name, input0, input1, output0, input0_data, input1_data, output_data):
  model = Model().Operation("REVERSE_EX", input0, input1).To(output0)
  example = Example({
      input0: input0_data,
      input1: input1_data,
      output0: output_data,
  }, model=model, name=name)

test(
    name="1d",
    input0=Input("input0", "TENSOR_FLOAT32", "{4}"),
    input1=Input("input1", "TENSOR_INT32", "{1}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{4}"),
    input0_data=[1, 2, 3, 4],
    input1_data=[0],
    output_data=[4, 3, 2, 1],
)

test(
    name="3d",
    input0=Input("input0", "TENSOR_FLOAT32", "{4, 3, 2}"),
    input1=Input("input1", "TENSOR_INT32", "{1}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{4, 3, 2}"),
    input0_data=[1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    input1_data=[1],
    output_data=[5,  6,  3,  4,  1,  2,  11, 12, 9,  10, 7,  8,
                 17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20],
)
