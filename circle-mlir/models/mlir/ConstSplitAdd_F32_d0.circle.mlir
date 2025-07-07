// To check Const-Split folding is working as expected
func.func @main_graph(%arg0: tensor<1x3x2x3xf32>) -> tensor<1x3x2x3xf32> attributes {
    input_names = ["i1"], output_names = ["o1"]}
{
  %0 = "Circle.pseudo_const"() {
    value = dense<[ 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                   18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
                   27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0]> : tensor<36xf32>
  } : () -> tensor<36xf32>
  %1 = "Circle.pseudo_const"() {value = dense<[2, 3, 2, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "Circle.reshape"(%0, %1) : (tensor<36xf32>, tensor<4xi32>) -> tensor<2x3x2x3xf32>
  // split_dim = 0
  %3 = "Circle.pseudo_const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>

  %4:2 = "Circle.split"(%3, %2) <{num_splits = 2 : i32}> :
    (tensor<i32>, tensor<2x3x2x3xf32>) -> (tensor<1x3x2x3xf32>, tensor<1x3x2x3xf32>)

  %5 = "Circle.add"(%arg0, %4#0) {fused_activation_function = "NONE"} :
    (tensor<1x3x2x3xf32>, tensor<1x3x2x3xf32>) -> tensor<1x3x2x3xf32>
  %6 = "Circle.add"(%5, %4#1) {fused_activation_function = "NONE"} :
    (tensor<1x3x2x3xf32>, tensor<1x3x2x3xf32>) -> tensor<1x3x2x3xf32>

  return %6 : tensor<1x3x2x3xf32>
}

// NOTE current test does not have validation to confirm folding is done or result is correct.
//      done some manual dump of generated circle file only having two or three 'Cirlce.add'.
// TODO make some test procedure (1) generate circle as-is (2) generate circle with folding
//      (3) compare circle-interpreter execution results of (1) and (2)
