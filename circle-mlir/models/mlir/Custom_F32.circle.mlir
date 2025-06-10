func.func @main_graph(%arg0: tensor<1x16x16xf32>) -> tensor<1x16x16xf32> attributes {
    input_names = ["onnx::QuantizeLinear_0"], output_names = ["5"]}
{
  %0 = "Circle.pseudo_const"() {
    value = dense<-0.0565590933> : tensor<f32>
  } : () -> tensor<f32>

  %1 = "Circle.pseudo_const"() {
    value = dense<-20> : tensor<i16>
  } : () -> tensor<i16>

  %2 = "Circle.custom"(%arg0, %0, %1) {
    custom_code = "ONNXQuantizeLinear",
    custom_option = #Circle<const_bytes : "0x540001030101010704022401">
  } : (tensor<1x16x16xf32>, tensor<f32>, tensor<i16>) -> tensor<1x16x16xi16>

  %3 = "Circle.custom"(%2, %0, %1) {
    custom_code = "ONNXDequantizeLinear",
    custom_option = #Circle<const_bytes : "0x540001030101010004022401">
  } : (tensor<1x16x16xi16>, tensor<f32>, tensor<i16>) -> tensor<1x16x16xf32>

  return %3 : tensor<1x16x16xf32>
}
