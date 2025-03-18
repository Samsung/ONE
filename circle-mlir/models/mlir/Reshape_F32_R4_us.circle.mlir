// Circle.reshape with unknown shape to validate shape inference
func.func @main_graph(%arg0: tensor<1x1x2x4xf32>) -> tensor<1x1x1x8xf32> attributes {
  input_names = ["onnx::Reshape_0"], output_names = ["2"]}
{
  %0 = "Circle.pseudo_const"() {value = dense<[1, 1, 1, 8]> : tensor<4xi32>} :
    () -> tensor<4xi32>
  %1 = "Circle.add"(%arg0, %arg0) {fused_activation_function = "NONE"} :
    (tensor<1x1x2x4xf32>, tensor<1x1x2x4xf32>) -> tensor<1x1x2x4xf32>
  %2 = "Circle.reshape"(%1, %0) :
    (tensor<1x1x2x4xf32>, tensor<4xi32>) -> tensor<1x1x1x?xf32>
  %3 = "Circle.add"(%2, %2) {fused_activation_function = "NONE"} :
    (tensor<1x1x1x?xf32>, tensor<1x1x1x?xf32>) -> tensor<1x1x1x8xf32>
  return %3 : tensor<1x1x1x8xf32>
}
