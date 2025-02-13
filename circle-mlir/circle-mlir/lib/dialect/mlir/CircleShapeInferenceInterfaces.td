#ifndef CIRCLE_SHAPE_INFERENCE_INTERFACE
#define CIRCLE_SHAPE_INFERENCE_INTERFACE

include "mlir/IR/OpBase.td"

def CIR_ShapeInferenceOpInterface : OpInterface<"CirShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}

#endif // CIRCLE_SHAPE_INFERENCE_INTERFACE
