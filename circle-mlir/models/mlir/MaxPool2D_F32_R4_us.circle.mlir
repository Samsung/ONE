module {
  func.func @main_graph(%arg0: tensor<1x2x16x9xf32>) -> tensor<1x?x?x?xf32> attributes {
      input_names = ["input"],
      output_names = ["output"]}
  {
    %0 = "Circle.max_pool_2d"(%arg0) {
      filter_height = 2 : i32,
      filter_width = 2 : i32,
      fused_activation_function = "NONE",
      padding = "VALID",
      stride_h = 2 : i32,
      stride_w = 2 : i32
    } : (tensor<1x2x16x9xf32>) -> tensor<1x?x?x?xf32>
    return %0 : tensor<1x?x?x?xf32>
  }
}
