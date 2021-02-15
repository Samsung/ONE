#include <string>

#include "tiramisu/tiramisu.h"

using namespace tiramisu;

void generate_code(const std::string &obj_filename)
{
    // Specify the name of the function that you want to create.
    tiramisu::init("arithmetic");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

#define SIZE0 (100)
    constant p0("N", expr((int32_t) SIZE0));

    var i("i", 0, p0), j("j", 0, p0), k("k", 0, p0);

    // Declare computations that represents the input buffers.  The actual
    // input buffers will be declared later.
    input A("A", {"i", "j"}, {SIZE0, SIZE0}, p_uint8);
    input B("B", {"i", "j"}, {SIZE0, SIZE0}, p_uint8);

    // Declare a computation to initialize the reduction.
    computation C_init("C_init", {i,j}, expr((uint8_t) 0));

    // Declare the reduction operation.  Do not provide any expression during declaration.
    computation C("C", {i,j,k}, p_uint8);
    // Note that the previous computation has an empty expression (because we can only use C in an expression after its declaration)
    C.set_expression(C(i, j, k - 1) + A(i, k) * B(k, j));

    // In this example, C does not read the value of C_init, but later
    // we indicate that C_init and C both are stored in the same buffer,
    // therefore C will read values written by C_init.
    // We are working on adding an operator for reduction to perform reduction
    // in a straight forward way.

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Tile both computations: C_init and C
    // This tiles the loop levels i and j and produces the loop levels by a 32x32 tile.
    // i0, j0, i1 and j1 where i0 is the outermost loop level and j1 is the innermost.

    var i0("i0"), j0("j0"), i1("i1"), j1("j1");
    C_init.tile(i, j, 32, 32, i0, j0, i1, j1);
    C.tile(i, j, 32, 32, i0, j0, i1, j1);

    // Parallelize the outermost loop level i0
    C.parallelize(i0);

    // Indicate that C is after C_init at the loop level j0 (this means,
    // they share the two outermost loops i0 and j0 and starting from j0 C
    // is ordered after C_init).
    C.after(C_init, j1);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // Declare the buffers.
    buffer b_A("b_A", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input);
    buffer b_B("b_B", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input);
    buffer b_C("b_C", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_output);

    // Map the computations to a buffer.
    A.store_in(&b_A);
    B.store_in(&b_B);

    // Store C_init[i,j] in b_C[i,j]
    C_init.store_in(&b_C, {i,j});
    // Store c_C[i,j,k] in b_C[i,j]
    C.store_in(&b_C, {i,j});
    // Note that both of the computations C_init and C store their
    // results in the buffer b_C.

    function *fct = global::get_implicit_function();
    fct->set_arguments({&b_A, &b_B, &b_C});
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

