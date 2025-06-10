module {
  func.func @main_graph(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x?x?x?xf32> attributes {
      input_names = ["input"],
      output_names = ["output"]}
  {
    %0 = "Circle.pseudo_const"() {value = dense<[2, 3]> :
      tensor<2xi32>} : () -> tensor<2xi32>
    %1 = "Circle.mean"(%arg0, %0) {keep_dims = true} :
      (tensor<1x2x4x4xf32>, tensor<2xi32>) -> tensor<1x?x?x?xf32>
    return %1 : tensor<1x?x?x?xf32>
  }
}
