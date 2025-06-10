func.func @main_graph() -> tensor<i64> attributes {
  input_names = [],
  output_names = ["output"]}
{
  %0 = "Circle.pseudo_const"() {
    value = dense<[123]> : tensor<1xi64>
  } : () -> tensor<1xi64>

  %1 = "Circle.pseudo_const"() {
    value = dense<[0]> : tensor<1xi32>
  } : () -> tensor<1xi32>

  %2 = "Circle.reduce_prod"(%0, %1) {keep_dims = false} :
    (tensor<1xi64>, tensor<1xi32>) -> tensor<i64>

  return %2 : tensor<i64>
}
