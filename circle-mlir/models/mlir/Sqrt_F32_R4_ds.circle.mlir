module {
  func.func @main_graph(%arg0: tensor<?x2x3x3xf32>) -> tensor<?x?x?x?xf32> attributes {
      input_names = ["input"],
      output_names = ["output"]}
  {
    %0 = "Circle.sqrt"(%arg0) : (tensor<?x2x3x3xf32>) -> tensor<?x?x?x?xf32>
    return %0 : tensor<?x?x?x?xf32>
  }
}
