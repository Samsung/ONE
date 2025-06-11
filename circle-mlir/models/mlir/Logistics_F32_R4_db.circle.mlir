// function with dynamic batch to validate '--dynamic_batch_to_single_batch' option
module {
  func.func @main_graph(%arg0: tensor<?x2x3x3xf32>) -> tensor<?x2x3x3xf32> attributes {
      input_names = ["input"],
      output_names = ["output"]}
  {
    %0 = "Circle.logistic"(%arg0) : (tensor<?x2x3x3xf32>) -> tensor<?x2x3x3xf32>
    return %0 : tensor<?x2x3x3xf32>
  }
}
