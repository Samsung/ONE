func.func @main_graph() -> tensor<2x3xi32> attributes {
  input_names = [],
  output_names = ["output"]
}
{
  %condition = "Circle.pseudo_const"() {
    value = dense<[[1, 1, 0], [0, 1, 0]]> : tensor<2x3xi1>
  } : () -> tensor<2x3xi1>

  %x = "Circle.pseudo_const"() {
    value = dense<[[-4572, 158053, 3566],
      [24901099, -177580, 44327]]> : tensor<2x3xi32>
  } : () -> tensor<2x3xi32>

  %y = "Circle.pseudo_const"() {
    value = dense<[[33505549, -1779503, 43055594],
      [355379045, -2436065, 2478764]]> : tensor<2x3xi32>
  } : () -> tensor<2x3xi32>

  %0 = "Circle.select_v2"(%condition, %x, %y) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>)
    -> tensor<2x3xi32>

  return %0 : tensor<2x3xi32>
}
