// Circle.reshape with 0 shape with input dim > 1
func.func @main_graph(%arg0: tensor<2x32x?x?xf32>) -> tensor<?x?x?xf32> attributes {
  input_names = ["input"], output_names = ["output"]}
{
  %0 = "Circle.pseudo_const"() {value = dense<[0, 4, -1]> : tensor<3xi32>} :
    () -> tensor<3xi32>
  %1 = "Circle.reshape"(%arg0, %0) :
    (tensor<2x32x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
