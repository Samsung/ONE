model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{2, 3}")
begins = Parameter("begins", "TENSOR_INT32", "{2}", [1, -1])
ends = Parameter("ends", "TENSOR_INT32", "{2}", [2, -4])
strides = Parameter("strides", "TENSOR_INT32", "{2}", [2, -1])
beginMask = Int32Scalar("beginMask", 0)
endMask = Int32Scalar("endMask", 0)
shrinkAxisMask = Int32Scalar("shrinkAxisMask", 0)

output = Output("output", "TENSOR_FLOAT32", "{1, 3}")

model = model.Operation("STRIDED_SLICE", i1, begins, ends, strides, beginMask, endMask, shrinkAxisMask).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3, 4, 5, 6]}

output0 = {output: # output 0
           [6, 5, 4]}

# Instantiate an example
Example((input0, output0))
