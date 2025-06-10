func.func @main_graph() -> tensor<2xi16> attributes {
input_names = [], output_names = ["output"]}
{
  %0 = "Circle.pseudo_const"() {
    value = dense<[-1, 1]> : tensor<2xi8>
  } : () -> tensor<2xi8>
  %1 = "Circle.cast"(%0) : (tensor<2xi8>) -> tensor<2xi16>

  return %1 : tensor<2xi16>
}
