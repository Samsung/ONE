def test(name, input0, input1, input2, output0, input0_data, input1_data, input2_data, output_data):
  model = Model().Operation("SELECT_V2_EX", input0, input1, input2).To(output0)
  example = Example({
      input0: input0_data,
      input1: input1_data,
      input2: input2_data,
      output0: output_data,
  }, model=model, name=name)

test(
    name="float",
    input0=Input("input0", "TENSOR_BOOL8", "{1, 1, 1, 4}"),
    input1=Input("input1", "TENSOR_FLOAT32", "{1, 1, 1, 4}"),
    input2=Input("input2", "TENSOR_FLOAT32", "{1, 1, 1, 4}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 1, 1, 4}"),
    input0_data=[True, False, True, False],
    input1_data=[1, 2, 3, 4],
    input2_data=[5, 6, 7, 8],
    output_data=[1, 6, 3, 8],
)

test(
    name="broadcast_1d_single_value",
    input0=Input("input0", "TENSOR_BOOL8", "{1}"),
    input1=Input("input1", "TENSOR_FLOAT32", "{1, 2, 2, 1}"),
    input2=Input("input2", "TENSOR_FLOAT32", "{1, 2, 2, 1}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 2, 2,1 }"),
    input0_data=[False],
    input1_data=[1, 2, 3, 4],
    input2_data=[5, 6, 7, 8],
    output_data=[5, 6, 7, 8],
)

test(
    name="broadcast_less_4d",
    input0=Input("input0", "TENSOR_BOOL8", "{1, 2}"),
    input1=Input("input1", "TENSOR_FLOAT32", "{1, 2, 2}"),
    input2=Input("input2", "TENSOR_FLOAT32", "{1, 2, 2}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 2, 2}"),
    input0_data=[False, True],
    input1_data=[1, 2, 3, 4],
    input2_data=[5, 6, 7, 8],
    output_data=[5, 2, 7, 4],
)

test(
    name="broadcast_2d_one",
    input0=Input("input0", "TENSOR_BOOL8", "{1, 1}"),
    input1=Input("input1", "TENSOR_FLOAT32", "{1, 1, 2, 2}"),
    input2=Input("input2", "TENSOR_FLOAT32", "{1, 1, 2, 2}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 1, 2, 2}"),
    input0_data=[False],
    input1_data=[1, 2, 3, 4],
    input2_data=[5, 6, 7, 8],
    output_data=[5, 6, 7, 8],
)

test(
    name="broadcast_2d_two",
    input0=Input("input0", "TENSOR_BOOL8", "{1, 2}"),
    input1=Input("input1", "TENSOR_FLOAT32", "{1, 2, 2}"),
    input2=Input("input2", "TENSOR_FLOAT32", "{1, 2, 1}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 2, 2}"),
    input0_data=[False, True],
    input1_data=[1, 2, 3, 4],
    input2_data=[5, 6], # 5 5 6 6
    output_data=[5, 2, 6, 4],
)
