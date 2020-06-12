import dynamic_tensor


# model
model = Model()

model_input1_shape = [3, 2, 2, 2]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input1_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput()

i2 = Input("op2", "TENSOR_FLOAT32","{2, 2}")
t1 = Internal("op3", "TENSOR_FLOAT32", "{}")
o1 = Output("op3", "TENSOR_FLOAT32", "{3, 2, 2, 2}")

model = model.Operation("SQUARED_DIFFERENCE_EX", test_node_input, i2).To(t1)
model = model.Operation("SQUARED_DIFFERENCE_EX", t1, i2).To(o1)

model_input1_data = [74.0, 63.0, 103.0, 88.0,
                      57.0, 9.0, 68.0, 20.0,
                      121.0, 38.0, 54.0, 119.0,
                      56.0, 106.0, 98.0, 98.0,
                      7.0, 89.0, 108.0, 104.0,
                      20.0, 81.0, 4.0, 124.0]
model_input2_data = [56.0, 115.0, 115.0, 116.0]

input0 = {
      dynamic_layer.getModelInput(): model_input1_data,   # input 1
      dynamic_layer.getShapeInput(): model_input1_shape,

      i2: model_input2_data # input 2
      }

output0 = {
      o1: [71824.0, 6702921.0, 841.0, 446224.0,
            3025.0, 123676640.0, 4384836.0, 82810000.0,
            17380560.0, 33802596.0, 13003236.0, 11449.0,
            3136.0, 1156.0, 30276.0, 43264.0,
            5499025.0, 314721.0, 4356.0, 784.0,
            1537600.0, 1083681.0, 148986432.0, 2704.0]
           }

# # Instantiate an example
Example((input0, output0))
