// shape inference with dynamic shape
func.func @main_graph(%arg0: tensor<1x?xi64>) -> tensor<?x?xf32> attributes {
  input_names = ["input"], output_names = ["output"]}
{
  %0 = "Circle.cast"(%arg0) : (tensor<1x?xi64>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
