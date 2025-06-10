func.func @main_graph
  (%arg0: tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32> attributes
    {input_names = ["input"], output_names = ["3"]}
{
  %0 = "Circle.pseudo_const"() {value = dense<[[[[-0.236339569, -6.820160e-01]]]]> :
    tensor<1x1x1x2xf32> } :
    () -> tensor<1x1x1x2xf32>
  %1 = "Circle.pseudo_const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} :
    () -> tensor<4xi32>
  %2 = "Circle.pseudo_const"() {value = dense<[-0.677482128, -0.602065206]> : tensor<2xf32>} :
    () -> tensor<2xf32>
  %3 = "Circle.pseudo_const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} :
    () -> tensor<4xi32>
  %4 = "Circle.transpose"(%arg0, %3) :
    (tensor<1x2x3x3xf32>, tensor<4xi32>) -> tensor<1x3x3x2xf32>
  %5 = "Circle.depthwise_conv_2d"(%4, %0, %2) {
    depth_multiplier = 1 : i32,
    dilation_h_factor = 1 : i32,
    dilation_w_factor = 1 : i32,
    fused_activation_function = "NONE",
    padding = "VALID",
    stride_h = 1 : i32,
    stride_w = 1 : i32} :
    (tensor<1x3x3x2xf32>, tensor<1x1x1x2xf32>, tensor<2xf32>) -> tensor<1x3x3x?xf32>
  %6 = "Circle.transpose"(%5, %1) :
    (tensor<1x3x3x?xf32>, tensor<4xi32>) -> tensor<1x2x3x3xf32>
  return %6 : tensor<1x2x3x3xf32>
}
