func.func @main_graph() -> tensor<6xf32> attributes {
  input_names = [],
  output_names = ["output"]}
{
  %0 = "Circle.pseudo_const"() {
    value = dense<[-0.077270925, 0.485836565, -0.0493729115,
      0.440191209, -0.493528903, 0.238003492]> : tensor<6xf32>
  } : () -> tensor<6xf32>

  %1 = "Circle.pseudo_const"() {
    value = dense<[[-0.45728302, 0.158099353, 0.3596766, -0.465268731],
      [0.249801099, -4.177580e-01, 0.434327662, 0.210706115],
      [0.337960482, -0.23828733, 0.387655854, -0.44274509],
      [0.335055649, -0.170279503, 0.43055594, -0.342647016],
      [0.355379045, -0.243609965, 0.247876644, 0.240414858],
      [0.395634234, 0.0652527213, -0.269038439, 0.258287847]]> : tensor<6x4xf32>
  } : () -> tensor<6x4xf32>

  %2 = "Circle.pseudo_const"() {
    value = dense<[1.16389668, -1.23830545, 1.15106285, -0.671000957]> : tensor<4xf32>
  } : () -> tensor<4xf32>

  %3 = "Circle.fully_connected"(%2, %1, %0) {
    asymmetric_quantize_inputs = false,
    fused_activation_function = "NONE",
    keep_num_dims = false,
    weights_format = "DEFAULT"
  } : (tensor<4xf32>, tensor<6x4xf32>, tensor<6xf32>) -> tensor<6xf32>

  return %3 : tensor<6xf32>
}
