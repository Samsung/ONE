func.func @main_graph() -> tensor<2xi64> attributes { output_names = ["output"]}
{
  %input = "Circle.pseudo_const"() {
    value = dense<[1, 80, 52, 52]> : tensor<4xi64>
  } : () -> tensor<4xi64>

  %begin = "Circle.pseudo_const"() {
    value = dense<[0]> : tensor<1xi32>
  } : () -> tensor<1xi32>

  %end = "Circle.pseudo_const"() {
    value = dense<[2]> : tensor<1xi32>
  } : () -> tensor<1xi32>

  %strides = "Circle.pseudo_const"() {
    value = dense<[1]> : tensor<1xi32>
  } : () -> tensor<1xi32>

  %0 = "Circle.strided_slice"(%input, %begin, %end, %strides) {
    onnx_node_name = "/Slice",
    begin_mask = 0 : i32,
    ellipsis_mask = 0 : i32,
    end_mask = 0 : i32,
    new_axis_mask = 0 : i32,
    shrink_axis_mask = 0 : i32
  } : (tensor<4xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>)
  -> tensor<2xi64>

  return %0 : tensor<2xi64>
}
