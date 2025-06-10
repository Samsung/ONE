func.func @main_graph(%arg0: tensor<4x4xf32>) -> tensor<4x6xf32> attributes {
  input_names = ["onnx::Gemm_0"],
  output_names = ["3"]}
{
  %0 = "Circle.pseudo_const"() {
    value = dense<[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]> : tensor<6xf32>
  } : () -> tensor<6xf32>

  %1 = "Circle.pseudo_const"() {
    value = dense<[
        [-0.220089614, 0.09057796, 0.0313191414, -0.278984904],
        [0.482074738, -0.298606098, -0.43428731, -0.470768571],
        [0.31505239, -1.574170e-01, -0.0900036692, 0.128537714],
        [0.0968894958, 0.415790677, -0.372064769, -0.0911480188],
        [-0.212634385, -0.406961322, -0.0873718261, 0.137704432],
        [-0.289212823, 0.407486856, -0.235535443, -0.16171068]
    ]> : tensor<6x4xf32>
  } : () -> tensor<6x4xf32>

  %2 = "Circle.fully_connected"(%arg0, %1, %0) {
    asymmetric_quantize_inputs = false,
    fused_activation_function = "NONE",
    keep_num_dims = false,
    weights_format = "DEFAULT"
  } : (tensor<4x4xf32>, tensor<6x4xf32>, tensor<6xf32>) -> tensor<4x6xf32>

  return %2 : tensor<4x6xf32>
}
