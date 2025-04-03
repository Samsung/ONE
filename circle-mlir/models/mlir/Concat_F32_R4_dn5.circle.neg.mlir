func.func @main_graph
  (%arg0: tensor<1x32x1x8xf32>, %arg1: tensor<1x32x1x8xf32>) -> tensor<1x64x1x8xf32> attributes {
    input_names = ["onnx::Concat_0", "onnx::Concat_1"], output_names = ["2"]}
{
  %0 = "Circle.concatenation"(%arg0, %arg1) {
    axis = -5 : i32,
    fused_activation_function = "NONE"
  } : (tensor<1x32x1x8xf32>, tensor<1x32x1x8xf32>) -> tensor<1x64x1x8xf32>

  return %0 : tensor<1x64x1x8xf32>
}
