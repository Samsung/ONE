# model
batch = 1
in_chans = 1
out_chans = 1
in_rows = 4
in_cols = 4
out_rows = 8
out_cols = 8
ker_rows = 3
ker_cols = 3
stride = 2
# pad is 0 (left: 0  right: 1 top: 0 bottom: 1)
input_table = [x for x in range(batch * in_rows * in_cols * in_chans)]
kernel_table = [x for x in range(out_chans * ker_rows * ker_cols * in_chans)]
out_table = [0 for x in range(batch * out_rows * out_cols * out_chans)]

for i in range(batch):
  for j in range(in_rows):
    for k in range(in_cols):
      for l in range(in_chans):
        out_row_origin = j * stride
        out_col_origin = k * stride
        input_value = input_table[((i * in_rows + j) * in_cols + k) * in_chans + l]

        for m in range(ker_rows):
          for n in range(ker_cols):
            for o in range(out_chans):
              out_row = out_row_origin + m
              out_col = out_col_origin + n
              if (out_row < out_rows) and (out_col < out_cols) and (out_row >= 0) and (out_col >= 0):
                kernel_value = kernel_table[((o * ker_rows + m) * ker_cols + n) * in_chans + l]
                out_table[((i * out_rows + out_row) * out_cols + out_col) * out_chans + o] += (input_value * kernel_value)

model = Model()
i0 = Input("op_shape", "TENSOR_INT32", "{4}")
weights = Parameter("ker", "TENSOR_FLOAT32", "{1, 3, 3, 1}", kernel_table)
i1 = Input("in", "TENSOR_FLOAT32", "{1, 4, 4, 1}" )
pad = Int32Scalar("pad_same", 1)
s_x = Int32Scalar("stride_x", 2)
s_y = Int32Scalar("stride_y", 2)
i2 = Output("op", "TENSOR_FLOAT32", "{1, 8, 8, 1}")
model = model.Operation("TRANSPOSE_CONV_EX", i0, weights, i1, pad, s_x, s_y).To(i2)

# Example 1. Input in operand 0,
input0 = {i0:  # output shape
          [1, 8, 8, 1],
          i1:  # input 0
          input_table}

output0 = {i2:  # output 0
           out_table}

# Instantiate an example
Example((input0, output0))
