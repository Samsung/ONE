func.func @main_graph()
-> tensor<?xi64> attributes {
  output_names = ["output"]}
{
  %data = "Circle.pseudo_const"() {
    value = dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi64>
  } : () -> tensor<8xi64>

  %begin = "Circle.pseudo_const"() {
    value = dense<[4]> : tensor<1xi32>
  } : () -> tensor<1xi32>
  
  %end = "Circle.pseudo_const"() {
    value = dense<[8]> : tensor<1xi32>
  } : () -> tensor<1xi32>

  %stride = "Circle.pseudo_const"() {
    value = dense<[1]> : tensor<1xi32>
  } : () -> tensor<1xi32>

  %0 = "Circle.strided_slice"(%data, %begin, %end, %stride) {
    begin_mask = 0 : i32, ellipsis_mask = 0 : i32,
    end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32
  } : (tensor<8xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>

  return %0 : tensor<?xi64>
}
