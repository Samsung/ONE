module {
  func.func @main_graph(%arg0: tensor<1x2x3x3xf32>, %arg1: tensor<1x2x3x3xf32>)
    -> tensor<1x2x3x3xf32> attributes {
      input_names = ["input_0", "input_1"],
      output_names = ["output_2"]}
  {
    %0 = Circle.add (%arg0, %arg1) {fused_activation_function = "NONE"} :
      (tensor<1x2x3x3xf32>, tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>
    return %0 : tensor<1x2x3x3xf32>
  }
}
