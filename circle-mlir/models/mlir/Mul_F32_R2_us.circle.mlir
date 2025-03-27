// output should be 6x3xf32
module {
  func.func @main_graph(%arg0: tensor<6x1xf32>) -> tensor<?x3xf32> attributes {
    input_names = ["input"],output_names = ["output"]
  }
  {
    %0 = "Circle.pseudo_const"() {
      value = dense<[[0.1, 0.2, 0.3]]> : tensor<1x3xf32>
    } : () -> tensor<1x3xf32>

    %1 = "Circle.mul"(%arg0, %0) {fused_activation_function = "NONE"} :
      (tensor<6x1xf32>, tensor<1x3xf32>) -> tensor<?x3xf32>
    return %1 : tensor<?x3xf32>
  }
}
