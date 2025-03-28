// Circle.strided_slice with unknown shape to validate shape inference
func.func @main_graph(%arg0: tensor<?x4xf32>) -> tensor<?x?xf32> attributes {
  input_names = ["input"], output_names = ["output"]}
{
  %0 = "Circle.pseudo_const"() {value = dense<0> : tensor<2xi32>} :() -> tensor<2xi32>
  %1 = "Circle.pseudo_const"() {value = dense<[0, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  %2 = "Circle.pseudo_const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
  %4 = "Circle.strided_slice"(%arg0, %0, %1, %2) {
    begin_mask = 0 : i32,
    ellipsis_mask = 0 : i32,
    end_mask = 0 : i32,
    new_axis_mask = 0 : i32,
    shrink_axis_mask = 0 : i32} :
    (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
