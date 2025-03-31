func.func @main_graph() -> tensor<4xf32> attributes {
input_names = [], output_names = ["output"]}
{
  %0 = "Circle.pseudo_const"() {
    value = dense<[-1, 0, 1, 64]> : tensor<4xi64>
  } : () -> tensor<4xi64>
  %1 = "Circle.cast"(%0) : (tensor<4xi64>) -> tensor<4xf32>

  return %1 : tensor<4xf32>
}
