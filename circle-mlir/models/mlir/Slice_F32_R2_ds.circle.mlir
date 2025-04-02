func.func @main_graph(%arg0: tensor<?x4xf32>) -> tensor<?x?xf32> attributes {
  input_names = ["input_data"],
  output_names = ["output"]}
{
  %starts = "Circle.pseudo_const"() {
    value = dense<[0]> : tensor<1xi32>
  } : () -> tensor<1xi32>

  %ends = "Circle.pseudo_const"() {
    value = dense<[2]> : tensor<1xi32>
  } : () -> tensor<1xi32>

  %axes = "Circle.pseudo_const"() {
    value = dense<[1]> : tensor<1xi32>
  } : () -> tensor<1xi32>

  %steps = "Circle.pseudo_const"() {
    value = dense<[1]> : tensor<1xi32>
  } : () -> tensor<1xi32>

  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) {
    onnx_node_name = "/Slice"
  } : (tensor<?x4xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>)
  -> tensor<?x?xf32>

  return %0 : tensor<?x?xf32>
}
