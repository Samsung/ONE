// Circle.resize_nearest_neighbor with unknown shape to validate shape inference
func.func @main_graph(%arg0: tensor<1x3x16x9xf32>) -> tensor<1x3x32x18xf32> attributes {
  input_names = ["input"], output_names = ["4"]}
{
  %0 = "Circle.pseudo_const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} :
    () -> tensor<4xi32>
  %1 = "Circle.pseudo_const"() {value = dense<[32, 18]> : tensor<2xi32>} :
    () -> tensor<2xi32>
  %2 = "Circle.pseudo_const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} :
    () -> tensor<4xi32>
  %3 = "Circle.transpose"(%arg0, %2) :
    (tensor<1x3x16x9xf32>, tensor<4xi32>) -> tensor<1x16x9x3xf32>
  %4 = "Circle.resize_nearest_neighbor"(%3, %1) {align_corners = false, half_pixel_centers = true} :
    (tensor<1x16x9x3xf32>, tensor<2xi32>) -> tensor<1x32x?x3xf32>
  %5 = "Circle.transpose"(%4, %0) :
    (tensor<1x32x?x3xf32>, tensor<4xi32>) -> tensor<1x3x32x18xf32>
  return %5 : tensor<1x3x32x18xf32>
}
