def test(name, lhs, rhs, output, adj_x, adj_y, lhs_data, rhs_data, output_data):
    model = Model().Operation("BATCH_MATMUL_EX", lhs, rhs, adj_x, adj_y).To(output)
    example = Example({
        lhs: lhs_data,
        rhs: rhs_data,
        output: output_data,
    }, model=model, name=name)

test(
    name='simple',
    lhs=Input("input0", "TENSOR_FLOAT32", "{1, 2, 3}"),
    rhs=Input("input1", "TENSOR_FLOAT32", "{1, 3, 4}"),
    lhs_data=[1, 2, 3, 4, 5, 6],
    rhs_data=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    adj_x=False,
    adj_y=False,
    output=Output("output0", "TENSOR_FLOAT32", "{1, 2, 4}"),
    output_data=[74, 80, 86, 92, 173, 188, 203, 218]
)

test(
    name='adj_y',
    lhs=Input("input0", "TENSOR_FLOAT32", "{1, 2, 3}"),
    rhs=Input("input1", "TENSOR_FLOAT32", "{1, 4, 3}"),
    lhs_data=[1, 2, 3, 4, 5, 6],
    rhs_data=[7, 11, 15, 8, 12, 16, 9, 13, 17, 10, 14, 18],
    adj_x=False,
    adj_y=True,
    output=Output("output0", "TENSOR_FLOAT32", "{1, 2, 4}"),
    output_data=[74, 80, 86, 92, 173, 188, 203, 218]
)

test(
    name='adj_x',
    lhs=Input("input0", "TENSOR_FLOAT32", "{1, 3, 2}"),
    rhs=Input("input1", "TENSOR_FLOAT32", "{1, 3, 4}"),
    lhs_data=[1, 4, 2, 5, 3, 6],
    rhs_data=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    adj_x=True,
    adj_y=False,
    output=Output("output0", "TENSOR_FLOAT32", "{1, 2, 4}"),
    output_data=[74, 80, 86, 92, 173, 188, 203, 218]
)

test(
    name='batch2',
    lhs=Input("input0", "TENSOR_FLOAT32", "{2, 2, 3}"),
    rhs=Input("input1", "TENSOR_FLOAT32", "{2, 3, 4}"),
    lhs_data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    rhs_data=[
        7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
    ],
    adj_x=False,
    adj_y=False,
    output=Output("output0", "TENSOR_FLOAT32", "{2, 2, 4}"),
    output_data=[
        74, 80, 86, 92, 173, 188, 203, 218,
        560, 584, 608, 632, 767, 800, 833, 866
    ]
)

test(
    name='broadcast',
    lhs=Input("input0", "TENSOR_FLOAT32", "{2, 2, 3}"),
    rhs=Input("input1", "TENSOR_FLOAT32", "{3, 4}"),
    lhs_data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    rhs_data=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    adj_x=False,
    adj_y=False,
    output=Output("output0", "TENSOR_FLOAT32", "{2, 2, 4}"),
    output_data=[
        74, 80, 86, 92, 173, 188, 203, 218, 272, 296,
        320, 344, 371, 404, 437, 470
    ]
)

test(
    name='broadcast_adj_x',
    lhs=Input("input0", "TENSOR_FLOAT32", "{2, 3, 2}"),
    rhs=Input("input1", "TENSOR_FLOAT32", "{3, 4}"),
    lhs_data=[1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12],
    rhs_data=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    adj_x=True,
    adj_y=False,
    output=Output("output0", "TENSOR_FLOAT32", "{2, 2, 4}"),
    output_data=[
        74, 80, 86, 92, 173, 188, 203, 218, 272, 296,
        320, 344, 371, 404, 437, 470
    ]
)

test(
    name='broadcast2_adj_xy',
    lhs=Input("input0", "TENSOR_FLOAT32", "{2, 1, 2, 3}"),
    rhs=Input("input1", "TENSOR_FLOAT32", "{3, 4, 2}"),
    lhs_data=[1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12],
    rhs_data=[
        7,  11, 8,  12, 9,  13, 10, 14, 15, 19, 16, 20,
        17, 21, 18, 22, 23, 27, 24, 28, 25, 29, 26, 30
    ],
    adj_x=True,
    adj_y=True,
    output=Output("output0", "TENSOR_FLOAT32", "{2, 3, 3, 4}"),
    output_data=[
        29,  32,  35,  38,  65,  72,  79,  86,  101,
        112, 123, 134, 53,  56,  59,  62,  121, 128,
        135, 142, 189, 200, 211, 222, 77,  80,  83,
        86,  177, 184, 191, 198, 277, 288, 299, 310,
        137, 152, 167, 182, 173, 192, 211, 230, 209,
        232, 255, 278, 257, 272, 287, 302, 325, 344,
        363, 382, 393, 416, 439, 462, 377, 392, 407,
        422, 477, 496, 515, 534, 577, 600, 623, 646
    ]
)
