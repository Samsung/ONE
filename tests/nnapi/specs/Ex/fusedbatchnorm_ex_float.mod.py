def test(name, x, scale, offset, mean, variance, is_training, data_format, epsilon, output,
         x_data, scale_data, offset_data, mean_data, variance_data, output_data ):
    model = Model().Operation("FUSED_BATCH_NORM_V3_EX",
                              x, scale, offset, mean, variance,              # inputs
                              is_training, data_format, epsilon).To(output)  # param
    example = Example({
        x: x_data,
        scale: scale_data,
        offset: offset_data,
        mean: mean_data,
        variance: variance_data,
        output: output_data,
    }, model=model, name=name)

test(
    name = 'fusedbatchnorm_1141',
    x = Input("input0", "TENSOR_FLOAT32", "{1,1,4,1}"),
    scale = Input("input1", "TENSOR_FLOAT32", "{1}"),
    offset = Input("input2", "TENSOR_FLOAT32", "{1}"),
    mean = Input("inputr3", "TENSOR_FLOAT32", "{1}"),
    variance = Input("input4", "TENSOR_FLOAT32", "{1}"),
    is_training = Parameter("is_training", "TENSOR_BOOL8", "{1}", [1]), # true
    data_format = Parameter("data_format", "TENSOR_QUANT8_ASYMM", "{4}, 1.0, 0",
                            [78, 72, 87, 67]), # NHWC: nnapi always assumes channel-last layout
    epsilon = Parameter("epsilon", "TENSOR_FLOAT32", "{1}", [0]),
    x_data=[0., 1., -1., 0. ],
    scale_data=[1.],
    offset_data=[0.],
    mean_data=[0.],
    variance_data=[1.],
    output=Output("output0", "TENSOR_FLOAT32", "{1,1,4,1}"),
    output_data=[0., 1.4142135381698608 , -1.4142135381698608, 0 ],
)
