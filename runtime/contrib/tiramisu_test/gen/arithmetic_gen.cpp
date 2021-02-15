#include <string>

#include "tiramisu/tiramisu.h"

using namespace tiramisu;

const int L = 4;
const int M = 100;
const int N = 8;

void generate_code(const std::string &obj_filename)
{
    // Specify the name of the function that you want to create.
    tiramisu::init("arithmetic");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    constant l_size("l_size", L), m_size("m_size", M), n_size("n_size", N) ;
    var l("l", 0, l_size), m("m", 0, m_size), n("n", 0, n_size);
    //inputs
    input A("A", {l, m}, p_uint8);
    input B("B", {m, n}, p_uint8);

    //Computations
    computation matmul_init("matmul_init", {l, n}, expr(cast(p_uint8, 0)));
    computation matmul("matmul", {l, n, m}, p_uint8);
    matmul.set_expression(matmul(l, n, m) + A(l, m)*B(m, n));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    matmul_init.then(matmul, n);
    matmul_init.parallelize(l);
    matmul.parallelize(l);


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer buf_A("A", {l_size, m_size}, p_uint8, a_input);
    buffer buf_B("B", {m_size, n_size}, p_uint8, a_input);
    buffer buf_matmul("matmul", {l_size, n_size}, p_uint8, a_output);

    //Store inputs
    A.store_in(&buf_A);
    B.store_in(&buf_B);

    //Store computations
    matmul_init.store_in(&buf_matmul, {l, n});
    matmul.store_in(&buf_matmul, {l, n});

    function *fct = global::get_implicit_function();
    fct->set_arguments({&buf_A, &buf_B, &buf_matmul});
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

