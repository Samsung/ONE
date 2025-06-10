func.func @main_graph() -> tensor<2xf32> attributes {
  input_names = [],
  output_names = ["output"]}
{
  %0 = "Circle.pseudo_const"() {
    value = dense<[-0.077270925, 0.485836565]> : tensor<2xf32>
  } : () -> tensor<2xf32>

  %1 = "Circle.cos"(%0) : (tensor<2xf32>) -> tensor<2xf32>

  return %1 : tensor<2xf32>
}
