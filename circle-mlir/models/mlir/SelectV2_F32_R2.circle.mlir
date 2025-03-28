func.func @main_graph() -> tensor<2x3xf32> attributes {
  input_names = [],
  output_names = ["output"]
}
{
  %condition = "Circle.pseudo_const"() {
    value = dense<[[1, 1, 0], [0, 1, 0]]> : tensor<2x3xi1>
  } : () -> tensor<2x3xi1>

  %x = "Circle.pseudo_const"() {
    value = dense<[[-4.572, 1.58053, 3.566],
      [2.4901099, -1.77580, 4.4327]]> : tensor<2x3xf32>
  } : () -> tensor<2x3xf32>

  %y = "Circle.pseudo_const"() {
    value = dense<[[3.3505549, -1.779503, 4.3055594],
      [3.55379045, -2.436065, 2.478764]]> : tensor<2x3xf32>
  } : () -> tensor<2x3xf32>

  %0 = "Circle.select_v2"(%condition, %x, %y) : (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>)
    -> tensor<2x3xf32>

  return %0 : tensor<2x3xf32>
}
