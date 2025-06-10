batch = 2
rows = 3
cols = 4
depth = 5
block_size_height = 2
block_size_width = 3
padding_size_height_top = 1
padding_size_height_bottom = 2
padding_size_width_left = 3
padding_size_width_right = 2
zero_value = 200
scale = 1.0
out_batch = batch * block_size_height * block_size_width
out_rows = (int)((rows + padding_size_height_top + padding_size_height_bottom) / block_size_height)
out_cols = (int)((cols + padding_size_width_left + padding_size_width_right) / block_size_width)

input_table = [x for x in range(batch * rows * cols * depth)]

output_table = [x for x in range(out_batch * out_rows * out_cols * depth)]
for b in range(batch):
  for h in range(rows + padding_size_height_top + padding_size_height_bottom):
    for w in range(cols + padding_size_width_left + padding_size_width_right):
      for d in range(depth):
        out_d = d;
        out_h = (int)(h / block_size_height);
        out_w = (int)(w / block_size_width);
        out_b = b + ((h % block_size_height) * block_size_width + w % block_size_width) * batch;

        if (h < padding_size_height_top) or (h >= (rows + padding_size_height_top)) or (w < padding_size_width_left) or (w >= (cols + padding_size_width_left)):
          output_table[out_b * out_rows * out_cols * depth + out_h * out_cols * depth + out_w * depth + out_d] = zero_value;
        else:
          output_table[out_b * out_rows * out_cols * depth + out_h * out_cols * depth + out_w * depth + out_d] = input_table[b * rows * cols * depth + (h - padding_size_height_top) * cols * depth + (w - padding_size_width_left) * depth + d];

i1 = Input("input", "TENSOR_QUANT8_ASYMM", "{%d, %d, %d, %d}, %d, %d" % (batch, rows, cols, depth, scale, zero_value))
block = Parameter("block_size", "TENSOR_INT32", "{2}", [block_size_height, block_size_width])
paddings = Parameter("paddings", "TENSOR_INT32", "{2, 2}", [padding_size_height_top, padding_size_height_bottom, padding_size_width_left, padding_size_width_right])
output = Output("output", "TENSOR_QUANT8_ASYMM", "{%d, %d, %d, %d}, %d, %d" % (out_batch, out_rows, out_cols, depth, scale, zero_value))

model = Model()
model = model.Operation("SPACE_TO_BATCH_ND", i1, block, paddings).To(output)


# Example 1. Input in operand 0,
input0 = {i1: # input 0
          input_table}

output0 = {output: # output 0
          output_table}

# Instantiate an example
Example((input0, output0))
