// Circle.strided_slice with unknown shape to validate shape inference
func.func @main_graph(%arg0: tensor<1x1x1x16xf32>) -> tensor<1x1x1x4xf32> attributes {
  input_names = ["input"], output_names = ["output"]}
{
  %0 = "Circle.pseudo_const"() {value = dense<0> : tensor<4xi32>} :() -> tensor<4xi32>
  %1 = "Circle.pseudo_const"() {value = dense<[1, 1, 1, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "Circle.pseudo_const"() {value = dense<1> : tensor<4xi32>} : () -> tensor<4xi32>
  %3 = "Circle.add"(%arg0, %arg0) {fused_activation_function = "NONE"} :
      (tensor<1x1x1x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
  %4 = "Circle.strided_slice"(%3, %0, %1, %2) {
    begin_mask = 0 : i32,
    ellipsis_mask = 0 : i32,
    end_mask = 0 : i32,
    new_axis_mask = 0 : i32,
    shrink_axis_mask = 0 : i32} :
    (tensor<1x1x1x16xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x1x1x?xf32>
  %5 = "Circle.add"(%4, %4) {fused_activation_function = "NONE"} :
    (tensor<1x1x1x?xf32>, tensor<1x1x1x?xf32>) -> tensor<1x1x1x4xf32>
  return %5 : tensor<1x1x1x4xf32>
}
