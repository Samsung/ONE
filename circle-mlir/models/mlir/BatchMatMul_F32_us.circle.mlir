module {
  func.func @main_graph(%arg0: tensor<1x3x5xf32>, %arg1: tensor<5x7xf32>)
    -> tensor<1x?x?xf32> attributes {
      input_names = ["input_0", "input_1"],
      output_names = ["output_2"]}
  {
    %0 = "Circle.batch_matmul"(%arg0, %arg1) :
      (tensor<1x3x5xf32>, tensor<5x7xf32>) -> tensor<1x?x?xf32>
    return %0 : tensor<1x?x?xf32>
  }
}
