model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{2, 3}")
begins = Parameter("begins", "TENSOR_INT32", "{2}", [0, 0])
ends = Parameter("ends", "TENSOR_INT32", "{2}", [2, 3])
strides = Parameter("strides", "TENSOR_INT32", "{2}", [2, 2])
beginMask = Int32Scalar("beginMask", 0)
endMask = Int32Scalar("endMask", 0)
shrinkAxisMask = Int32Scalar("shrinkAxisMask", 0)

output = Output("output", "TENSOR_FLOAT32", "{1, 2}")

model = model.Operation("STRIDED_SLICE", i1, begins, ends, strides, beginMask, endMask, shrinkAxisMask).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 2.0, 3.0,
		   4.0, 5.0, 6.0]}

output0 = {output: # output 0
           [1.0, 3.0]}

# Instantiate an example
Example((input0, output0))
