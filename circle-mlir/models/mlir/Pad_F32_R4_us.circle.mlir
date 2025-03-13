// Circle.pad with unknown shape to validate shape inference
func.func @main_graph(%arg0: tensor<1x2x3x3xf32>) -> tensor<1x2x10x6xf32> attributes {
  input_names = ["input"], output_names = ["1"]}
{
  %0 = "Circle.pseudo_const"() {value = dense<[[0, 0], [0, 0], [3, 4], [1, 2]]> : tensor<4x2xi32>} :
    () -> tensor<4x2xi32>
  %1 = "Circle.add"(%arg0, %arg0) {fused_activation_function = "NONE"} :
    (tensor<1x2x3x3xf32>, tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
  %2 = "Circle.pad"(%1, %0) :
    (tensor<1x2x3x3xf32>, tensor<4x2xi32>) -> tensor<1x2x?x?xf32>
  %3 = "Circle.add"(%2, %2) {fused_activation_function = "NONE"} :
    (tensor<1x2x?x?xf32>, tensor<1x2x?x?xf32>) -> tensor<1x2x10x6xf32>
  return %3 : tensor<1x2x10x6xf32>
}
