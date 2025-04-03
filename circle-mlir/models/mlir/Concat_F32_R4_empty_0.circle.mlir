func.func @main_graph
  (%arg0: tensor<1x32x1x8xf32>, %arg1: tensor<1x0x1x8xf32>) -> tensor<1x32x1x8xf32> attributes {
    input_names = ["input_0", "input_1"], output_names = ["output"]}
{
  %0 = "Circle.concatenation"(%arg0, %arg1) {
    axis = 1 : i32,
    fused_activation_function = "NONE"
  } : (tensor<1x32x1x8xf32>, tensor<1x0x1x8xf32>) -> tensor<1x32x1x8xf32>

  return %0 : tensor<1x32x1x8xf32>
}
