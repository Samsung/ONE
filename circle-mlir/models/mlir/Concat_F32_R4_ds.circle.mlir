// Circle.concatenation with unknown shape to validate dynamic shape inference
func.func @main_graph(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x?x3x4xf32>) -> tensor<?x?x?x?xf32>
  attributes {input_names = ["input_0", "input_1"], output_names = ["output"]}
{
  %1 = "Circle.concatenation"(%arg0, %arg1) { axis = 1 : i32, fused_activation_function = "NONE" } :
      (tensor<1x2x3x4xf32>, tensor<1x?x3x4xf32>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}
