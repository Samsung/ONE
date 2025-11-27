# model
model = Model()
start = Input("start", "TENSOR_FLOAT32", "{}")
limit = Input("limit", "TENSOR_FLOAT32", "{}")
delta = Input("delta", "TENSOR_FLOAT32", "{}")
out = Output("output", "TENSOR_FLOAT32", "{3}")
model = model.Operation("RANGE_EX", start, limit, delta).To(out)

input0 = {start: [10],
           limit: [3],
           delta: [-3]}

output0 = {out: # output 0
           [10.0, 7.0, 4.0]}
# Instantiate an example
Example((input0,output0))
