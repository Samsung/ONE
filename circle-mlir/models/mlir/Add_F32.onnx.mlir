module attributes {
  llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  llvm.target_triple = "x86_64-unknown-linux-gnu"}
{
  func.func @main_graph(%arg0: tensor<1x2x3x3xf32>, %arg1: tensor<1x2x3x3xf32>)
    -> tensor<1x2x3x3xf32> attributes {
      input_names = ["0", "1"],
      output_names = ["2"]}
  {
    %0 = "onnx.Add"(%arg0, %arg1) {
      onnx_node_name = "Add_0"
    } : (tensor<1x2x3x3xf32>, tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>

    return %0 : tensor<1x2x3x3xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
