// Circle.concatenation with unknown shape to validate shape inference
func.func @main_graph(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> tensor<1x4x3x4xf32>
  attributes {input_names = ["input_0", "input_1"], output_names = ["output"]}
{
  %0 = "Circle.add"(%arg0, %arg1) {fused_activation_function = "NONE"} :
      (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %1 = "Circle.concatenation"(%0, %0) { axis = 1 : i32, fused_activation_function = "NONE" } :
      (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x?x3x4xf32>
  %2 = "Circle.add"(%1, %1) {fused_activation_function = "NONE"} :
      (tensor<1x?x3x4xf32>, tensor<1x?x3x4xf32>) -> tensor<1x4x3x4xf32>
  return %2 : tensor<1x4x3x4xf32>
}
