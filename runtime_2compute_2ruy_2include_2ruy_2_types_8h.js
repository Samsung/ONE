var runtime_2compute_2ruy_2include_2ruy_2_types_8h =
[
    [ "nnfw::ruy::PaddingValues", "structnnfw_1_1ruy_1_1_padding_values.html", "structnnfw_1_1ruy_1_1_padding_values" ],
    [ "nnfw::ruy::ConvParams", "structnnfw_1_1ruy_1_1_conv_params.html", "structnnfw_1_1ruy_1_1_conv_params" ],
    [ "nnfw::ruy::FullyConnectedParams", "structnnfw_1_1ruy_1_1_fully_connected_params.html", "structnnfw_1_1ruy_1_1_fully_connected_params" ],
    [ "nnfw::ruy::MatrixParams< Scalar >", "structnnfw_1_1ruy_1_1_matrix_params.html", "structnnfw_1_1ruy_1_1_matrix_params" ],
    [ "nnfw::ruy::GemmParams< AccumScalar, DstScalar, quantization_flavor >", "structnnfw_1_1ruy_1_1_gemm_params.html", "structnnfw_1_1ruy_1_1_gemm_params" ],
    [ "CachePolicy", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#aa1b02e2e4161dc332b8f81e16d93532d", [
      [ "kNeverCache", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#aa1b02e2e4161dc332b8f81e16d93532da034316273cd939ada300353e91737a83", null ],
      [ "kCacheIfLargeSpeedup", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#aa1b02e2e4161dc332b8f81e16d93532dab8c7e8866dc5309079f1ee98258639e9", null ],
      [ "kAlwaysCache", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#aa1b02e2e4161dc332b8f81e16d93532da0d3d239e220c4289e5dc0bf3b1687422", null ]
    ] ],
    [ "FusedActivationFunctionType", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#adc4ac5ec1dbd6adeb7a34920bf80fb88", [
      [ "kNone", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#adc4ac5ec1dbd6adeb7a34920bf80fb88a35c3ace1970663a16e5c65baa5941b13", null ],
      [ "kRelu6", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#adc4ac5ec1dbd6adeb7a34920bf80fb88a4c7f8cb4a4aefd7ac6a89fa1b747233d", null ],
      [ "kRelu1", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#adc4ac5ec1dbd6adeb7a34920bf80fb88ae0a0000573fa134f9ee575b47d839ee4", null ],
      [ "kRelu", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#adc4ac5ec1dbd6adeb7a34920bf80fb88a067892a9ea619b2c378e06000c9763af", null ],
      [ "kTanh", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#adc4ac5ec1dbd6adeb7a34920bf80fb88a4918cf5d849692c2bed918bb9e948630", null ],
      [ "kSigmoid", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#adc4ac5ec1dbd6adeb7a34920bf80fb88ac00732693e14261bf9c2a4612a7f9bf9", null ]
    ] ],
    [ "Order", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#a53e614d61e4a56cdc05773ce61d9179b", [
      [ "kColMajor", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#a53e614d61e4a56cdc05773ce61d9179baf093768651925b22c4d08e33641f38f1", null ],
      [ "kRowMajor", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#a53e614d61e4a56cdc05773ce61d9179ba1ebc644af759b214a70279505401a0b9", null ]
    ] ],
    [ "PaddingType", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#a61f6822bae7b9115886e65067857cac4", [
      [ "kNone", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#a61f6822bae7b9115886e65067857cac4a35c3ace1970663a16e5c65baa5941b13", null ],
      [ "kSame", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#a61f6822bae7b9115886e65067857cac4ad523906435f67e94367d4fab5b9380da", null ],
      [ "kValid", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#a61f6822bae7b9115886e65067857cac4a4d3576c37e6f03700bad4345238fffa0", null ]
    ] ],
    [ "QuantizationFlavor", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#a719af35c91a72f276105e09b0a40395d", [
      [ "kFloatingPoint", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#a719af35c91a72f276105e09b0a40395da6b5725464f6e8ab0a2cdf49a29a0f4a9", null ],
      [ "kIntegerWithUniformMultiplier", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#a719af35c91a72f276105e09b0a40395dac11557b4d4fc196c43b4fa7eab201144", null ],
      [ "kIntegerWithPerRowMultiplier", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#a719af35c91a72f276105e09b0a40395da227d5cf4d9445848d0940bb4dd05ee37", null ]
    ] ],
    [ "DefaultCachePolicy", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#a88bfa77bd1460059a8934b213d1cdb55", null ],
    [ "ValidateGemmParams", "runtime_2compute_2ruy_2include_2ruy_2_types_8h.html#a695d6cbce17540c90564d4c369083181", null ]
];