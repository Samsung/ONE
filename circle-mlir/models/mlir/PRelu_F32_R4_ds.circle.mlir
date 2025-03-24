// Circle.prelu with dynamic input shape
func.func @main_graph(%arg0: tensor<?x2x3x3xf32>) -> tensor<?x?x?x?xf32> attributes {
  input_names = ["input"], output_names = ["output"]}
{
  %0 = "Circle.pseudo_const"() {value = dense<2.500000e-01> : tensor<1xf32>} :
    () -> tensor<1xf32>
  %1 = "Circle.prelu"(%arg0, %0) :
    (tensor<?x2x3x3xf32>, tensor<1xf32>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}
