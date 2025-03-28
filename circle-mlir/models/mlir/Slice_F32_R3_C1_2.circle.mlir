func.func @main_graph(%arg0: tensor<3x2x3xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>)
-> tensor<3x2x?xf32> attributes {
  input_names = ["input_data", "starts", "ends"],
  output_names = ["output"]}
{
  %axes = "Circle.pseudo_const"() {
    value = dense<[-1]> : tensor<1xi32>
  } : () -> tensor<1xi32>

  %steps = "Circle.pseudo_const"() {
    value = dense<[1]> : tensor<1xi32>
  } : () -> tensor<1xi32>

  %0 = "onnx.Slice"(%arg0, %arg1, %arg2, %axes, %steps) {
    onnx_node_name = "/Slice"
  } : (tensor<3x2x3xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>)
  -> tensor<3x2x?xf32>

  return %0 : tensor<3x2x?xf32>
}
