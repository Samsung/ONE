# model
model = Model()
start = Input("start", "TENSOR_FLOAT32", "{}")
limit = Input("limit", "TENSOR_FLOAT32", "{}") 
delta = Input("delta", "TENSOR_FLOAT32", "{}") 
out = Output("output", "TENSOR_FLOAT32", "{8}") 
model = model.Operation("RANGE_EX", start, limit, delta).To(out)

input0 = {start: [1],
           limit: [5],
           delta: [0.5]}

output0 = {out: # output 0
           [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]}
# Instantiate an example
Example((input0,output0))
