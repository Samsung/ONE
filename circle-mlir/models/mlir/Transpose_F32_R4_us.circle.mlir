// Circle.transpose with unknown shape to validate shape inference
func.func @main_graph(%arg0: tensor<1x3x6x4xf32>) -> tensor<1x6x4x3xf32> attributes {
  input_names = ["onnx::Transpose_0"], output_names = ["1"]}
{
  %0 = "Circle.pseudo_const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "Circle.add"(%arg0, %arg0) {fused_activation_function = "NONE"} :
    (tensor<1x3x6x4xf32>, tensor<1x3x6x4xf32>) -> tensor<1x3x6x4xf32>
  %2 = "Circle.transpose"(%1, %0) : (tensor<1x3x6x4xf32>, tensor<4xi32>) -> tensor<1x?x?x?xf32>
  %3 = "Circle.add"(%2, %2) {fused_activation_function = "NONE"} :
    (tensor<1x?x?x?xf32>, tensor<1x?x?x?xf32>) -> tensor<1x6x4x3xf32>
  return %3 : tensor<1x6x4x3xf32>
}
