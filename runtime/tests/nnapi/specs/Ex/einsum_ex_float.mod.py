def test(name, lhs, rhs, equation, output, lhs_data, rhs_data, output_data):
    model = Model().Operation("EINSUM_EX", lhs, rhs, equation).To(output)
    example = Example({
        lhs: lhs_data,
        rhs: rhs_data,
        output: output_data,
    }, model=model, name=name)

test(
    name = 'matmul_2x2_2',
    lhs = Input("input0", "TENSOR_FLOAT32", "{2, 3}"),
    rhs = Input("input1", "TENSOR_FLOAT32", "{3, 2}"),
    equation = Parameter("eq", "TENSOR_QUANT8_ASYMM", "{9}, 1.0, 0",
        [105, 107, 44, 107, 106, 45, 62, 105, 106]), # ik,kj->ij
    lhs_data=[0., 1., 2., 3., 4., 5.],
    rhs_data=[0., 3., 1., 4., 2., 5.],
    output=Output("output0", "TENSOR_FLOAT32", "{2, 2}"),
    output_data=[5., 14., 14., 50.]
)

# abc,cde->abde
dim_a = 2
dim_b = 2
dim_c = 4
dim_d = 3
dim_e = 3

lhs_value = [x for x in range(dim_a * dim_b * dim_c)]
rhs_value = [x for x in range(dim_c * dim_d * dim_e)]
result_value = [0 for x in range(dim_a * dim_b * dim_d * dim_e)]

for a in range(dim_a):
    for b in range(dim_b):
        for d in range(dim_d):
            for e in range(dim_e):
                result_index = e + dim_e * (d + dim_d * (b + dim_b * a))

                for c in range(dim_c):
                    lhs_index = c + dim_c * (b + dim_b * a)
                    rhs_index = e + dim_e * (d + dim_d * c)
                    result_value[result_index] = result_value[result_index] + lhs_value[lhs_index] * rhs_value[rhs_index]

test(
    name = 'matmul_3x3_4',
    lhs = Input("input0", "TENSOR_FLOAT32", "{2, 2, 4}"),
    rhs = Input("input1", "TENSOR_FLOAT32", "{4, 3, 3}"),
    equation = Parameter("eq", "TENSOR_QUANT8_ASYMM", "{13}, 1.0, 0",
        [97, 98, 99, 44, 99, 100, 101, 45, 62, 97, 98, 100, 101]), # abc,cde->abde
    lhs_data=lhs_value,
    rhs_data=rhs_value,
    output=Output("output0", "TENSOR_FLOAT32", "{2, 2, 3, 3}"),
    output_data=result_value
)

# abc,cd->abd
dim_a = 2
dim_b = 3
dim_c = 4
dim_d = 3

lhs_value = [x for x in range(dim_a * dim_b * dim_c)]
rhs_value = [x for x in range(dim_c * dim_d)]
result_value = [0 for x in range(dim_a * dim_b * dim_d)]

for a in range(dim_a):
    for b in range(dim_b):
        for d in range(dim_d):
            result_index = d + dim_d * (b + dim_b * a)

            for c in range(dim_c):
                lhs_index = c + dim_c * (b + dim_b * a)
                rhs_index = d + dim_d * c
                result_value[result_index] = result_value[result_index] + lhs_value[lhs_index] * rhs_value[rhs_index]

test(
    name = 'matmul_3x2_3',
    lhs = Input("input0", "TENSOR_FLOAT32", "{2, 3, 4}"),
    rhs = Input("input1", "TENSOR_FLOAT32", "{4, 3}"),
    equation = Parameter("eq", "TENSOR_QUANT8_ASYMM", "{11}, 1.0, 0",
        [97, 98, 99, 44, 99, 100, 45, 62, 97, 98, 100]), # abc,cd->abd
    lhs_data=lhs_value,
    rhs_data=rhs_value,
    output=Output("output0", "TENSOR_FLOAT32", "{2, 3, 3}"),
    output_data=result_value
)


# abcd,adbe->acbe
dim_a = 2
dim_b = 3
dim_c = 4
dim_d = 2
dim_e = 4

lhs_value = [x for x in range(dim_a * dim_b * dim_c * dim_d)]
rhs_value = [x for x in range(dim_a * dim_d * dim_b * dim_e)]
result_value = [0 for x in range(dim_a * dim_c * dim_b * dim_e)]

for a in range(dim_a):
    for c in range(dim_c):
        for b in range(dim_b):
            for e in range(dim_e):
                result_index = e + dim_e * (b + dim_b * (c + dim_c * a))

                for d in range(dim_d):
                    lhs_index = d + dim_d * (c + dim_c * (b + dim_b * a))
                    rhs_index = e + dim_e * (b + dim_b * (d + dim_d * a))
                    result_value[result_index] = result_value[result_index] + lhs_value[lhs_index] * rhs_value[rhs_index]

test(
    name = 'matmul_4x4_4',
    lhs = Input("input0", "TENSOR_FLOAT32", "{2, 3, 4, 2}"),
    rhs = Input("input1", "TENSOR_FLOAT32", "{2, 2, 3, 4}"),
    equation = Parameter("eq", "TENSOR_QUANT8_ASYMM", "{15}, 1.0, 0",
        [97, 98, 99, 100, 44, 97, 100, 98, 101, 45, 62, 97, 99, 98, 101]), # abcd,adbe->acbe
    lhs_data=lhs_value,
    rhs_data=rhs_value,
    output=Output("output0", "TENSOR_FLOAT32", "{2, 4, 3, 4}"),
    output_data=result_value
)

# abcd,aecd->aceb
dim_a = 2
dim_b = 3
dim_c = 2
dim_d = 4
dim_e = 3

lhs_value = [x for x in range(dim_a * dim_b * dim_c * dim_d)]
rhs_value = [x for x in range(dim_a * dim_e * dim_c * dim_d)]
result_value = [0 for x in range(dim_a * dim_c * dim_e * dim_b)]

for a in range(dim_a):
    for c in range(dim_c):
        for b in range(dim_e):
            for e in range(dim_b):
                result_index = b + dim_b * (e + dim_e * (c + dim_c * a))

                for d in range(dim_d):
                    lhs_index = d + dim_d * (c + dim_c * (b + dim_b * a))
                    rhs_index = d + dim_d * (c + dim_c * (e + dim_e * a))
                    result_value[result_index] = result_value[result_index] + lhs_value[lhs_index] * rhs_value[rhs_index]

test(
    name = 'matmul_4x4_4_2',
    lhs = Input("input0", "TENSOR_FLOAT32", "{2, 3, 2, 4}"),
    rhs = Input("input1", "TENSOR_FLOAT32", "{2, 3, 2, 4}"),
    equation = Parameter("eq", "TENSOR_QUANT8_ASYMM", "{15}, 1.0, 0",
        [97, 98, 99, 100, 44, 97, 101, 99, 100, 45, 62, 97, 99, 101, 98]), # abcd,aecd->aceb
    lhs_data=lhs_value,
    rhs_data=rhs_value,
    output=Output("output0", "TENSOR_FLOAT32", "{2, 2, 3, 3}"),
    output_data=result_value
)
