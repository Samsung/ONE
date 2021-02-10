#include <string>

#include "tiramisu/tiramisu.h"

using namespace tiramisu;

void generate_code(const std::string &obj_filename)
{
    // Specify the name of the function that you want to create.
    tiramisu::init("arithmetic");

    constexpr int N = 384;

    var inputx("inputx", 0, N);
    var inputyz("inputyz", 0, N*3);

    var left_branch_idx("left_idx", 0, N);
    var right_branch_idx("right_idx", 0, N);
    var merge_idx("merge_idx", 0, N);

    input x({inputx}, p_float32);
    input y({inputyz}, p_float32);
    input z({inputyz}, p_float32);

    buffer x_b("x", {N}, p_float32, a_input);
    buffer y_b("y", {N*3}, p_float32, a_input);
    buffer z_b("z", {N*3}, p_float32, a_input);

    x.store_in(&x_b);
    y.store_in(&y_b);
    z.store_in(&z_b);

    constexpr int unroll_factor = 4;
//    constexpr int vector_factor = 4;  // vectorization did not work properly

    expr add1 = y(left_branch_idx) + z(left_branch_idx);
    expr logistic1 = expr(float(1)) / (expr(o_expo, -add1) + 1);
    computation left_branch({left_branch_idx}, logistic1);
    buffer left_b("left", {N}, p_float32, a_temporary);
    left_branch.store_in(&left_b);
//    left_branch.vectorize(left_branch_idx, vector_factor);
    left_branch.unroll(left_branch_idx, unroll_factor);

    expr add2 = y(right_branch_idx + N) + z(right_branch_idx + N);
    expr logistic2 = expr(float(1)) / (expr(o_expo, -add2) + 1);
    expr mul2 = y(right_branch_idx + N*2) * logistic2;
    expr add3 = z(right_branch_idx + N*2) + mul2;
    computation right_branch({right_branch_idx}, add3);
    buffer right_b("right", {N}, p_float32, a_temporary);
    right_branch.store_in(&right_b);
//    right_branch.vectorize(right_branch_idx, vector_factor);
    right_branch.unroll(right_branch_idx, unroll_factor);

    expr sub = expr(1.f) - left_branch(merge_idx);
//    expr tanh = expr(o_tanh, right_branch(merge_idx));
    expr tanh = (expr(float(1)) / (expr(o_expo, -right_branch(merge_idx)) + 1) + 1)/2.0f;
    expr mul3 = sub * tanh;
    expr mul1 = left_branch(merge_idx) * x(merge_idx);
    expr add4 = mul1 + mul3;
    computation merge_branches({merge_idx}, add4);
    buffer output_b("output", {N}, p_float32, a_output);
    merge_branches.store_in(&output_b);
//    merge_branches.vectorize(merge_idx, vector_factor);
    merge_branches.unroll(merge_idx, unroll_factor);

    left_branch.then(right_branch, computation::root)
               .then(merge_branches, computation::root);

    function *fct = global::get_implicit_function();
//    fct->set_arguments({&x_b, &y_b, &z_b, &add4_buf});
    fct->set_arguments({&x_b, &y_b, &z_b, &output_b});
//    fct->lift_dist_comps();
    fct->gen_time_space_domain();
    fct->gen_isl_ast();
    fct->gen_halide_stmt();

    Halide::Target::OS os = Halide::Target::Linux;
    Halide::Target::Arch arch = Halide::Target::ARM;
    int bits = 32;

    fct->gen_halide_obj(obj_filename, os, arch, bits);
}

int main(int argc, char **argv){
  generate_code("arithmetic.o");
}

