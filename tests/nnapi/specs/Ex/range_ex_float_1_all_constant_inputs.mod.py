# model
model = Model()
shape = Input("lhs", "TENSOR_INT32", "{2}")
start = Parameter("start", "TENSOR_FLOAT32", "{}", [1])
limit = Parameter("limit", "TENSOR_FLOAT32", "{}", [5])
delta = Parameter("delta", "TENSOR_FLOAT32", "{}", [0.5])
range_out = Internal("range_out", "TENSOR_FLOAT32", "{8}")
out = Output("output", "TENSOR_FLOAT32", "{1, 8}")
model = model.Operation("RANGE_EX", start, limit, delta).To(range_out)
model = model.Operation("RESHAPE", range_out, shape).To(out)

input0 = {shape: [1, 8]}

output0 = {out: # output 0
           [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]}
# Instantiate an example
Example((input0,output0))
