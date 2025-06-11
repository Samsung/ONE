// Circle.prelu with unknown shape to validate shape inference
func.func @main_graph(%arg0: tensor<1x2x3x3xf32>) -> tensor<1x2x?x?xf32> attributes {
  input_names = ["input"], output_names = ["1"]}
{
  %0 = "Circle.relu"(%arg0) : (tensor<1x2x3x3xf32>) -> tensor<1x2x?x?xf32>
  return %0 : tensor<1x2x?x?xf32>
}
