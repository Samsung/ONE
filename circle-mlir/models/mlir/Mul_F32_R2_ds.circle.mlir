// Circle.mul with unknown shape to validate dynamic shape inference
module {
  func.func @main_graph(%arg0: tensor<?x1xf32>) -> tensor<?x?xf32> attributes {
    input_names = ["input"],output_names = ["output"]
  }
  {
    %0 = "Circle.pseudo_const"() {
      value = dense<[[0.1, 0.2, 0.3]]> : tensor<1x3xf32>
    } : () -> tensor<1x3xf32>

    %1 = "Circle.mul"(%arg0, %0) {fused_activation_function = "NONE"} :
      (tensor<?x1xf32>, tensor<1x3xf32>) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
}
