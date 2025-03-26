/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// from tensorflow/compiler/mlir/lite/ir/tfl_ops.td

#ifndef CIRCLE_OPS
#define CIRCLE_OPS

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/CircleOpInterfaces.td"
include "mlir/CircleShapeInferenceInterfaces.td"
include "mlir/CircleOpEnums.td"

//===----------------------------------------------------------------------===//
// Derived shape attribute class.
//===----------------------------------------------------------------------===//

class DerivedCircleTypeAttr<code body, code convert> :
  DerivedAttr<"circle::TensorType", body, convert>;

// CIR Runtime op trait predicate.
class CIR_RuntimePredOpTrait<string desc, Pred pred> :
    GenInternalOpTrait<"CIRRuntimeOpTrait"> {
  Pred cirRuntimePredicate = pred;
  string cirRuntimeDescription = desc;
}

class CIR_OperandsHaveSameShapesOrBroadcastableShape<
    list<int> indices, int max_bcast_rank> :
  CIR_RuntimePredOpTrait<"operands do not have the same shape or "
      "broadcastable shapes within the rank " # max_bcast_rank,
    CPred<"Circle::VerifyOperandsHaveSameShapesOrBroadcastableShape("
            "$_op, llvm::ArrayRef<unsigned>({" # !interleave(indices, ", ") #
            "}), " # max_bcast_rank # ")">>;

// Returns true if the n-th operand has unknown rank or at least rank m.
class CIR_OperandHasAtleastRank<int n, int m> :
  PredOpTrait<"operand " # n # " is " # m # "-D",
    Or<[CPred<"$_op.getOperand(" # n # ").getType().isa<UnrankedTensorType>()">,
      CPred<"$_op.getOperand(" # n #
        ").getType().cast<ShapedType>().getRank() >= " # m>]>>;

// CIR Runtime type predicate.
class CIR_RuntimeType<TypeConstraint t> {
  Pred circRuntimeTypePredicate = t.predicate;
  string cirRuntimeTypeDescription = t.summary;
}

class CIR_TensorOf<list<Type> allowedRuntimeTypes,
                   list<Type> allowedOpTypes = [AnyType]> :
  TensorOf<allowedOpTypes>, CIR_RuntimeType<TensorOf<allowedRuntimeTypes>> {
  // Set the summary equal to that representing the runtime types.
  let summary = TensorOf<allowedRuntimeTypes>.summary;
}

class CIR_TensorOfOrNone<list<Type> allowedRuntimeTypes, string description = "",
                         list<Type> allowedOpTypes = [AnyType]> :
  AnyTypeOf<[CIR_TensorOf<allowedOpTypes>, NoneType], description>,
  CIR_RuntimeType<AnyTypeOf<[CIR_TensorOf<allowedRuntimeTypes>, NoneType]>>;

class CIR_VariadicTensorOf<list<Type> allowedRuntimeTypes,
                   list<Type> allowedOpTypes = [AnyType]> :
  Variadic<TensorOf<allowedOpTypes>>,
  CIR_RuntimeType<Variadic<TensorOf<allowedRuntimeTypes>>>;

def CIR_Int32Or64 : SignlessIntOfWidths<[32, 64]>;

def CIR_BoolTensor : CIR_TensorOf<[I1]>;
def CIR_FpTensor : CIR_TensorOf<[F32]>;
def CIR_I32OrI64Tensor : CIR_TensorOf<[CIR_Int32Or64]>;
def CIR_I32Tensor : CIR_TensorOf<[I32]>;

class CIR_0DTensorOf<list<Type> allowedRuntimeTypes,
                     list<Type> allowedOpTypes = [AnyType]> :
  0DTensorOf<allowedOpTypes>, CIR_RuntimeType<TensorOf<allowedRuntimeTypes>>;
class CIR_1DTensorOf<list<Type> allowedRuntimeTypes,
                     list<Type> allowedOpTypes = [AnyType]> :
  1DTensorOf<allowedOpTypes>, CIR_RuntimeType<TensorOf<allowedRuntimeTypes>>;

class CIR_1DTensorOfOrNone<list<Type> allowedRuntimeTypes, string description = "",
                         list<Type> allowedOpTypes = [AnyType]> :
  AnyTypeOf<[TensorOf<allowedOpTypes>, NoneType], description>,
  CIR_RuntimeType<AnyTypeOf<[CIR_1DTensorOf<allowedRuntimeTypes>, NoneType]>>;

//===----------------------------------------------------------------------===//
// Rank/Shape helpers.
//===----------------------------------------------------------------------===//

class CIR_OperandIsUnrankedPred<int n> :
  CPred<"$_op.getOperand(" # n # ").getType().isa<UnrankedTensorType>()">;

// TODO: Some of these could be generalized and/or moved to more general
// location.
// Returns true if the n-th operand has unknown rank or has rank m.
class CIR_OperandHasRank<int n, int m> :
  PredOpTrait<"operand " # n # " is " # m # "-D",
    Or<[CIR_OperandIsUnrankedPred<n>,
      CPred<"$_op.getOperand(" # n #
      ").getType().cast<ShapedType>().getRank() == " # m>]>>;

class CIR_TFTypesWithSameBits<int i, int j, int num> :
  And<[CPred<"getElementTypeOrSelf($_op.getResult(" # i # ")).isUnsignedInteger(" # num # ")">,
       CPred<"getElementTypeOrSelf($_op.getOperand(" # j # ")).isUnsignedInteger(" # num # ")">]>;

class CIR_TFOperandTypesWithSameBits<int i, int j, int num> :
  And<[
    Or<[/*CPred<"getElementTypeOrSelf($_op.getOperand(" # i # ")).isa<mlir::TF::Quint" # num # "Type>()">,*/
        CPred<"getElementTypeOrSelf($_op.getOperand(" # i # ")).isUnsignedInteger(" # num # ")">]>,
    Or<[/*CPred<"getElementTypeOrSelf($_op.getOperand(" # j # ")).isa<mlir::TF::Quint" # num # "Type>()">,*/
        CPred<"getElementTypeOrSelf($_op.getOperand(" # j # ")).isUnsignedInteger(" # num # ")">]>]>;

class CIR_OperandHasRankAtMostPred<int n, int m> :
  Or<[CIR_OperandIsUnrankedPred<n>,
    CPred<"$_op.getOperand(" # n #
    ").getType().cast<ShapedType>().getRank() <= " # m>]>;

// True if operand n is ranked and has a rank > dim.
class CIR_OperandIsRankedAndHasDimPred<int n, int dim> : And<[
  CPred<"$_op.getOperand(" # n # ").getType().isa<RankedTensorType>()">,
  CPred<"$_op.getOperand(" # n # ").getType().cast<ShapedType>().getRank() > "
  # dim>]>;

// Returns true if the n-th operand is ranked and has a dimension length <=
// size at the rank dim.
class CIR_OperandDimIsAtMost<int n, int dim, int size> : And<[
  CIR_OperandIsRankedAndHasDimPred<n, dim>,
  CPred<"$_op.getOperand(" # n # ").getType().cast<ShapedType>()"
      ".getShape()[" # dim # " ] <= " # size>]>;

class CIR_OperandRankEquals1DimOfOperand<int x, int y> :
  PredOpTrait<"operand " # x # "'s rank equals operand " # y # "'s size",
    Or<[CIR_OperandIsUnrankedPred<x>,
        CIR_OperandIsUnrankedPred<y>,
        CPred<"!$_op.getOperand(" # y #
          ").getType().cast<ShapedType>().hasStaticShape()">,
        CPred<"$_op.getOperand(" # x #
          ").getType().cast<ShapedType>().getRank() == "
          "$_op.getOperand(" # y #
          ").getType().cast<ShapedType>().getShape()[0]">]>>;

class CIR_OperandHasRankAtMost<int n, int m> :
  PredOpTrait<"operand " # n # " is at most " # m # "-D",
    CIR_OperandHasRankAtMostPred<n, m>>;

class CIR_OperandHasRankAtLeast<int n, int m> :
  PredOpTrait<"operand " # n # " is at least " # m # "-D",
    Or<[CIR_OperandIsUnrankedPred<n>,
      CPred<"$_op.getOperand(" # n #
      ").getType().cast<ShapedType>().getRank() >= " # m>]>>;

// Ensures the array attribute's size is within the given maximum size.
class CIR_ArrayMaxCount<int n> : AttrConstraint<
    CPred<"$_self.isa<ArrayAttr>() && $_self.cast<ArrayAttr>().size() <= " # n>,
    "whose size is at most " # n>;

// This is a quantization-aware version of TCresVTEtIsSameAsOp
class CIR_TCresVTEtIsSameAsOp<int i, int j> : And<[
  TCOpResIsShapedTypePred<i, j>,
  Or<[
    TCresVTEtIsSameAsOpBase<i, j>,
    CIR_TFTypesWithSameBits<i, j, 8>/* TODO enable,
    And<[
      SubstLeaves<"$_self", "getElementTypeOrSelf($_op.getOperand(" # j # "))",
        quant_QuantizedType.predicate>,
      CPred<"quant::QuantizedType::castToStorageType("
                "getElementTypeOrSelf($_op.getResult(" # i # "))) == "
            "quant::QuantizedType::castToStorageType("
                "getElementTypeOrSelf($_op.getOperand(" # j # ")))">]>*/]>]>;

// This is a quantization-aware version of TCopVTEtAreSameAt
class CIR_TCopVTEtAreSameAt<int i, int j, int num=8> : Or<[
  TCopVTEtAreSameAt<[i, j]>,
  CIR_TFOperandTypesWithSameBits<i, j, num>/*,
  And<[
    SubstLeaves<"$_self", "getElementTypeOrSelf($_op.getOperand(" # j # "))",
      quant_QuantizedType.predicate>,
    CPred<"quant::QuantizedType::castToStorageType("
              "getElementTypeOrSelf($_op.getOperand(" # i # "))) == "
          "quant::QuantizedType::castToStorageType("
              "getElementTypeOrSelf($_op.getOperand(" # j # ")))">]>*/]>;

def CIR_SameFirstOperandAndFirstResultElementType :
  PredOpTrait<"values and output must have same element type",
              CIR_TCresVTEtIsSameAsOp<0, 0>>;

//===----------------------------------------------------------------------===//
// CIR op common constraints.
//===----------------------------------------------------------------------===//

class OperandsSameElementTypeConstraintBase<string op> :
  PredOpTrait<op # " operands have same element type",
    Or<[
      TCopVTEtIsSameAs<0, 1>/*,
      // Two operands' values are both quantized and their type have the same
      // underlying storage type.
      And<[
        SubstLeaves<"$_self", "getElementTypeOrSelf($_op.getOperand(0))",
          quant_QuantizedType.predicate>,
        CPred<"quant::QuantizedType::castToStorageType("
                  "getElementTypeOrSelf($_op.getOperand(0))) == "
              "quant::QuantizedType::castToStorageType("
                  "getElementTypeOrSelf($_op.getOperand(1)))">]>*/]>>;

// This is a constraint for most of the binary ops, e.g., add, mul, div, etc.
// Binary ops lhs & rhs should have the same value type, and is capable to
// compare quantization types as well.
def BinaryOpSameElementTypeConstraint :
  OperandsSameElementTypeConstraintBase<"binary op">;

// This is a constraint for most of the comparison ops, e.g., equal, not_equal,
// greater, greater_equal, less, etc. Comparison ops lhs & rhs should have the
// same value type, and is capable to compare quantization types as well.
def ComparisonOpSameElementTypeConstraint :
  OperandsSameElementTypeConstraintBase<"comparison op">;

//===----------------------------------------------------------------------===//
// CIR common builders.
//===----------------------------------------------------------------------===//

def CIR_BroadcastableBinaryBuilder :
  OpBuilder<(ins "Value":$lhs, "Value":$rhs),
  [{
    auto resultType =
      OpTrait::util::getBroadcastedType(lhs.getType(), rhs.getType());
    if (!resultType)
      mlir::emitError($_state.location, "non-broadcastable operands");
    $_state.addOperands({lhs, rhs});
    $_state.types.push_back(resultType);
  }]>;

class CIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<CIR_Dialect, mnemonic, !listconcat(traits,
      [DeclareOpInterfaceMethods<CIR_RuntimeVerification>])> {
  // FlatBuffer generation specific information.
  // -------------------------------------------
  // When generating the FlatBuffer output some operations have
  // Options (as defined in the schema). These options are effectively
  // the attributes of the operations (e.g., what padding is to be used
  // for a pooling operator). Not all operations have Options and some
  // operations share Options. The following attributes indicate whether
  // the operation has Options in the serialized FlatBuffer.

  // Whether the Circle operator has options in the schema representation.
  bit hasOptions = 0b0;

  // Use to specify a custom options type for Circle operators where
  // the option's name does not match the Cirlce operator's name.
  // If no customOption is specified then <name>Options is used if the op
  // hasOptions.
  string customOption = ?;
}

// NOTE 3'rd argument int index is removed, add when needed
class CIR_ConvOp<string mnemonic, string opSummary,
                 list<Trait> additional_traits = []> :
    CIR_Op<mnemonic,[Pure,
                     // TODO enable AccumulatorUniformScale<2, 0, 1>,
                     // TODO enable AffineQuantizedOpInterface,
                     // TODO enable AffineOpCoefficient<index, 1>,
                     // TODO enable QuantizableResult,
                     CIR_SparseOp] # additional_traits> {
  let summary = opSummary # " operator";

  let description = [{
    Performs convolution operation on inputs.

    Inputs:
      `inputs[0]`: required: the input activation tensor
      `inputs[1]`: required: the filter weight tensor
      `inputs[2]`: optional: the bias tensor
  }];

  let results = (outs CIR_TensorOf<[F32/*TODO enable, QI8, QUI8, QI16*/]>:$output);

  let hasOptions = 0b1;
}

//===----------------------------------------------------------------------===//
// CIR op definitions.
//===----------------------------------------------------------------------===//
def CIR_AddOp : CIR_Op<"add", [
    CIR_RuntimePredOpTrait<"Operands do not have valid shapes",
      CPred<"Circle::VerifyAddOpShapeConstraints(llvm::cast<AddOp>($_op))">>,
    ResultsBroadcastableShape,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    Pure,
    Commutative,
    // TODO enable QuantizableResult,
    ]> {
  let summary = "Addition operator";

  let description = [{
    Element-wise addition operation.
  }];

  let arguments = (
    // TODO add more dtypes
    ins CIR_TensorOf<[F32, I32, I64]>:$lhs,
    CIR_TensorOf<[F32, I32, I64]>:$rhs,
    CIR_AFAttr:$fused_activation_function);

  let results = (outs CIR_TensorOf<[F32, I32, I64]>:$output);

  let hasFolder = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];

  let hasOptions = 1;
}

def CIR_ArgMaxOp : CIR_Op<"arg_max", [
    // QuantizableResult,
    Pure]> {
  let summary = "ArgMax operator";

  let description = [{
    Returns the index with the largest value across dimensions of a tensor.
  }];

  let arguments = (
    ins CIR_TensorOf<[I1, F32, I32, I8, UI8/*, QI8, QUI8*/]>:$input,
    CIR_I32OrI64Tensor:$dim
  );

  let results = (outs
    CIR_I32OrI64Tensor:$output
  );

  let hasOptions = 1;

  DerivedCircleTypeAttr output_type = DerivedCircleTypeAttr<[{
    return getResult().getType().cast<TensorType>().getElementType().
        cast<IntegerType>().getWidth() > 32 ? circle::TensorType_INT64 :
            circle::TensorType_INT32;
    }], [{
      TypeAttr::get(getResult().getType().cast<TensorType>().getElementType())
    }]>;
}

def CIR_ConstOp : Op<CIR_Dialect, "pseudo_const", [ConstantLike, Pure,
    FirstAttrDerivedResultType,
    // TODO enable QuantizableResult,
    DeclareOpInterfaceMethods<CIR_RuntimeVerification>]> {
  let summary = "Constant pseudo op.";

  let description = [{
    Represents a constant value in Circle dialect. This is not an
    actual operation and it will be lowered to buffer instead.

    The op is allowed to have all the same type of attributes as tf.Const does
    (e.g., opaque TF attributes are allowed).
  }];

  let arguments = (ins ElementsAttr:$value);

  let results = (outs AnyTensor:$output);

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<(ins "TypedAttr":$value),
    [{
      $_state.addAttribute("value", value);
      $_state.addTypes(value.getType());
    }]>
  ];

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(Attribute value, Type type);
  }];
}

def CIR_Conv2DOp : CIR_ConvOp<"conv_2d", "Convolution", /*TODO enable 0,*/
      [DeclareOpInterfaceMethods<InferTypeOpInterface>,
       DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
       /*TODO enable,
       DynamicRangeQuantizedOpInterface*/]> {
  let arguments = (
    ins CIR_TensorOf<[F32/*TODO enable, QI8, QUI8, QI16*/]>:$input,
    CIR_TensorOf<[F32/*TODO enable QI4, QI8, QUI8*/]>:$filter,
    CIR_1DTensorOfOrNone<[F32, I32, I64]>:$bias,
    I32Attr:$dilation_h_factor,
    I32Attr:$dilation_w_factor,
    CIR_AFAttr:$fused_activation_function,
    CIR_PaddingAttr:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w
  );

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    // AffineQuantizedOpInterface:
    int GetChannelDimIndex() { return 0; }
    int GetQuantizationDimIndex() { return 0; }
    // SparseOpInterface:
    std::vector<int> GetSparseOperands() { return {1}; }
    std::vector<std::vector<int>> GetFloatBlockSize() { return {}; }
    std::vector<std::vector<int>> GetQuantizedBlockSize() { return {}; }

    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);

    // DynamicRangeQuantizedOpInterface:
    bool GetDynamicRangeQuantKernelSupport() { return true; }
    std::vector<int> GetQuantizableOperandIndices() { return {1}; }
  }];
}

def CIR_CosOp: CIR_Op<"cos", [
    Pure,
    /*CIR_SameOperandsAndResultTypeResolveRef*/
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    SameOperandsAndResultShape]> {
  let summary = "Cosine operator";

  let description = [{
    Computes element-wise Cosine of input
  }];

  let arguments = (ins CIR_FpTensor:$x);

  let results = (outs CIR_FpTensor:$y);

  let hasFolder = 1;

  // TODO enable TF::ArraysAreCastCompatible
  //let extraClassDeclaration = [{
  //  // Returns whether the return types are compatible.
  //  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  //    return TF::ArraysAreCastCompatible(l, r);
  //  }
  //}];
}

def CIR_CumsumOp: CIR_Op<"cumsum", [
    Pure,
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    CIR_OperandHasRank<1, 0>]> {
  let summary = "Cumsum operator";

  let description = [{
    Compute the cumulative sum of the tensor x along axis.
  }];

  let arguments = (
    ins CIR_TensorOf<[F32, I32, I64]>:$input,
    CIR_I32Tensor:$axis,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$exclusive,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$reverse
  );

  let results = (outs CIR_TensorOf<[F32, I32, I64]>:$output);

  let hasOptions = 1;
}

def CIR_CustomOp : Op<CIR_Dialect, "custom", [
  DeclareOpInterfaceMethods<CIR_RuntimeVerification>,
  DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>
  ]> {
  let summary = "Custom op";

  let description = [{
    A generic op for any Circle custom operation.

    input: A list of inputs in the original op.
    custom_code: A string used to identify which exactly this op is, which
                 corresponds to operator_codes.custom_code in the flatbuffer.
    custom_option: a holder to save the op attributes in bytes fashion.
    output: A list of outputs in the original op.
  }];

  let arguments = (ins
    Variadic<CIR_TensorOfOrNone<[AnyType]>>:$input,
    StrAttr:$custom_code,
    CIR_ConstBytesAttr:$custom_option
  );
  let results = (outs Variadic<AnyTensor>:$output);

  let hasVerifier = 1;
}

def CIR_DivOp : CIR_Op<"div", [
    // TODO(fengliuai): NoQuantizableResult is only correct for int8
    // quantization. update to handle Uint8 quantization.
    BinaryOpSameElementTypeConstraint,
    CIR_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 5>,
    ResultsBroadcastableShape,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    Pure]> {
  let summary = "Division operator";

  let description = [{
    Element-wise division operation.
  }];

  let arguments = (
      ins CIR_TensorOf<[F32, I32/*, QUI8*/]>:$lhs,
      CIR_TensorOf<[F32, I32/*, QUI8*/]>:$rhs,
      CIR_AFAttr:$fused_activation_function);

  let results = (outs CIR_TensorOf<[F32, I32/*, QUI8*/]>:$output);

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];

  let hasOptions = 1;

  let hasFolder = 1;
}

// TODO(jpienaar): Update post discussion on semantics of FC OP.
def CIR_FullyConnectedOp : CIR_Op<"fully_connected", [
    Pure, /*AccumulatorUniformScale<2, 0, 1>,*/
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    /*AffineQuantizedOpInterface,*/
    /*AffineOpCoefficient<-1, 1>,*/
    /*CIR_SparseOp,*/
    /*QuantizableResult,*/
    /*DynamicRangeQuantizedOpInterface*/]> {
  let summary = "Fully connected op";

  let arguments = (ins
    CIR_TensorOf<[F32/*, QI8, QUI8, QI16, QUI16*/]>:$input,
    CIR_TensorOf<[F32/*, QI8, QUI8, QI16*/]>:$filter,
    CIR_TensorOfOrNone<[F32/*, QI32, QUI32*/]>:$bias,

    CIR_AFAttr:$fused_activation_function,
    CIR_FullyConnectedOptionsWeightFormatAttr:$weights_format,
    BoolAttr:$keep_num_dims,
    // Used in post-training dynamic range quantization. If the value is true,
    // input activations are asymmetrically quantized.
    OptionalAttr<BoolAttr>:$asymmetric_quantize_inputs
  );

  // Depending on the weights format, this op can have one or two outputs.
  let results = (outs
    CIR_VariadicTensorOf<[F32/*, QI8, QUI8, QI16, QUI16*/]>:$output
  );

  let hasVerifier = 1;

  let hasOptions = 1;

  let hasCanonicalizer = 1;

  let hasFolder = 1;

  let extraClassDeclaration = [{
    // AffineQuantizedOpInterface:
    int GetChannelDimIndex() { return 0; }
    int GetQuantizationDimIndex() { return -1; }
    // SparseOpInterface:
    std::vector<int> GetSparseOperands() { return {1}; }
    std::vector<std::vector<int>> GetFloatBlockSize() { return {{1, 4}}; }
    std::vector<std::vector<int>> GetQuantizedBlockSize() { return {{1, 16}}; }
    // DynamicRangeQuantizedOpInterface:
    bool RequireAsymmetricQuantizeInputsAttr() { return true; }
    bool GetDynamicRangeQuantKernelSupport() { return true; }
    std::vector<int> GetQuantizableOperandIndices() { return {1}; }
  }];
}

def CIR_LeakyReluOp: CIR_Op<"leaky_relu", [
    SameOperandsAndResultShape,
    // QuantizableResult,
    Pure,
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>]> {
  let summary = "Leaky Relu operator";

  let description = [{
    Element-wise Leaky ReLU operator
      x -> x >= 0 ? x : (alpha * x)
  }];

  let arguments = (
    ins CIR_TensorOf<[F32/*, QUI8, QI8, TFL_Quint8, QI16*/]>:$input,
    // Slope of the activation function at x < 0.
    F32Attr:$alpha
  );

  let results = (outs CIR_TensorOf<[F32/*, QUI8, QI8, TFL_Quint8, QI16*/]>:$output);

  let hasOptions = 0b1;
}

def CIR_MaximumOp : CIR_Op<"maximum", [
    ResultsBroadcastableShape,
    Pure,
    /*QuantizableResult,*/
    CIR_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 5>,
    Commutative/*,*/
    /*SameOperandsAndResultsScale*/]> {
  let summary = "Max operator";
  let description = [{
    Element-wise max operation.
  }];

  let arguments = (
    ins CIR_TensorOf<[F32, CIR_Int32Or64/*, QI8, QUI8, QI16*/]>:$lhs,
    CIR_TensorOf<[F32, CIR_Int32Or64/*, QI8, QUI8, QI16*/]>:$rhs
  );

  let results = (outs
    CIR_TensorOf<[F32, CIR_Int32Or64/*, QI8, QUI8, QI16*/]>:$max
  );

  let builders = [CIR_BroadcastableBinaryBuilder];

  let hasOptions = 0;
}

def CIR_MeanOp : CIR_Op<"mean", [
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    /*QuantizableResult,*/
    Pure]> {
  let summary = "Mean operator";

  let description = [{
    Computes the mean of elements across dimensions of a tensor.
    Reduces input_tensor along the dimensions given in axis.
    Unless keepdims is true, the rank of the tensor is reduced by 1 for
    each entry in axis. If keepdims is true, the reduced dimensions are retained
    with length 1.
  }];

  let arguments = (ins
    CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, UI8, QI16*/]>:$input,
    CIR_TensorOf<[I32, I64]>:$axis,
    BoolAttr:$keep_dims
  );

  let results = (outs
    CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, UI8, QI16*/]>:$output);

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def CIR_MinimumOp : CIR_Op<"minimum", [
    ResultsBroadcastableShape,
    Pure,
    CIR_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 5>,
    Commutative/*,*/
    /*QuantizableResult,*/
    /*SameOperandsAndResultsScale*/]> {
  let summary = "Min operator";
  let description = [{
    Element-wise min operation.
  }];

  let arguments = (
    ins CIR_TensorOf<[F32, CIR_Int32Or64/*, QI8, QUI8, QI16*/]>:$lhs,
    CIR_TensorOf<[F32, CIR_Int32Or64/*, QI8, QUI8, QI16*/]>:$rhs
  );

  let results = (outs
    CIR_TensorOf<[F32, CIR_Int32Or64/*, QI8, QUI8, QI16*/]>:$min
  );

  let builders = [CIR_BroadcastableBinaryBuilder];

  let hasOptions = 0;
}

def CIR_MulOp : CIR_Op<"mul", [
    ResultsBroadcastableShape,
    Pure,
    Commutative,
    /*QuantizableResult,*/
    BinaryOpSameElementTypeConstraint,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    CIR_RuntimePredOpTrait<"Operands do not have valid shapes",
      CPred<"Circle::VerifyMulOpShapeConstraints(llvm::cast<MulOp>($_op))">>]> {
  let summary = "Multiplication operator";

  let description = [{
    Element-wise multiplication operation.
  }];

  let arguments = (
    ins CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, QI16, Complex<F<32>>*/]>:$lhs,
    CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, QI16, Complex<F<32>>*/]>:$rhs,
    CIR_AFAttr:$fused_activation_function);

  let results = (outs CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, QI16, Complex<F<32>>*/]>:$output);

  let hasFolder = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];

  let hasOptions = 1;
}

def CIR_NoValueOp : Op<CIR_Dialect, "no_value", [ConstantLike, Pure]> {
  let summary = "constant representing no value.";

  let description = [{
    No value constant op.
  }];

  let arguments = (ins UnitAttr:$value);

  let results = (outs NoneType:$none_val);

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(Attribute value, Type type);
  }];
}

def CIR_PadOp : CIR_Op<"pad", [
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    /*SameOperandsAndResultsScale,*/
    CIR_OperandHasRankAtMost<0, 5>,
    CIR_OperandHasRank<1, 2>,
    CIR_OperandRankEquals1DimOfOperand<0, 1>,
    /*QuantizableResult,*/
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    PredOpTrait<"the first dim size of the padding argument must be at most 5",
      Or<[CIR_OperandIsUnrankedPred<1>,
          CIR_OperandDimIsAtMost<1, 0, 5>]>>]> {
  let summary = "Padding operator";

  let description = [{
    This operation pads a `input` with zeros according to the `paddings` you
    specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is
    the rank of `input`. For each dimension D of `input`, `paddings[D, 0]`
    indicates how many zeros to add before the contents of `input` in that
    dimension, and `paddings[D, 1]` indicates how many zeros to add after the
    contents of `input` in that dimension.

    The padded size of each dimension D of the output is:

      `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

    For example:

    ```
    # 't' is [[1, 1], [2, 2]]
    # 'paddings' is [[1, 1], [2, 2]]
    # rank of 't' is 2
    pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                          [0, 0, 1, 1, 0, 0]
                          [0, 0, 2, 2, 0, 0]
                          [0, 0, 0, 0, 0, 0]]
    ```
  }];

  let arguments = (ins CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, TFL_Quint8, QI16*/]>:$input,
    CIR_I32OrI64Tensor:$padding);

  let results = (outs CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, TFL_Quint8, QI16*/]>:$output);

  let hasOptions = 1;

  let hasFolder = 1;
}

def CIR_PReluOp : CIR_Op<"prelu", [
    Pure,
    //QuantizableResult,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    ResultsBroadcastableShape,
    CIR_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>,
    BinaryOpSameElementTypeConstraint,
    PredOpTrait<"input and output must have the same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    /*AffineQuantizedOpInterface, AffineOpCoefficient<-1, 1>*/]> {
  let summary = "Parameterized Relu operator";

  let description = [{
    Parameterized Relu operator
      x -> x >= 0 ? x : (alpha * x)
    where alpha is a trainable tensor.
    input and alpha should be the same size as input or be broadcastable.
  }];

  let arguments = (
    ins CIR_TensorOf<[F32/*, QI8, QUI8, TFL_Quint8*/]>:$input,
    CIR_TensorOf<[F32/*, QI8, QUI8, TFL_Quint8*/]>:$alpha
  );

  let results = (outs CIR_TensorOf<[F32/*, QI8, QUI8, TFL_Quint8*/]>:$output);

  let hasVerifier = 1;

  /*
  let extraClassDeclaration = [{
    // AffineQuantizedOpInterface:
    int GetChannelDimIndex() { return 0; }
    int GetQuantizationDimIndex() { return -1; }
  }];
  */
}

def CIR_ReduceMaxOp: CIR_Op<"reduce_max", [
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    /*QuantizableResult,*/
    /*SameOperandsAndResultsScale*/]> {
  let summary = "Max-reduction operator";

  let description = [{
    Computes the max reduction along the specified axes
  }];

  let arguments = (ins
    CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, CIR_Quint8, QI16*/]>:$input,
    CIR_I32Tensor:$axes,
    BoolAttr:$keep_dims
  );

  let results = (outs
    CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, CIR_Quint8, QI16*/]>:$output);

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def CIR_ReduceProdOp: CIR_Op<"reduce_prod", [
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    /*QuantizableResult,*/
    Pure]> {
  let summary = "Prod-reduction operator";

  let description = [{
    Computes the product along the specified axes
  }];

  let arguments = (ins
    CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, TFL_Quint8, QI16*/]>:$input,
    CIR_I32Tensor:$axes,
    BoolAttr:$keep_dims
  );

  let results = (outs
    CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, TFL_Quint8, QI16*/]>:$output);

  let hasFolder = 1;

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def CIR_ReluOp: CIR_Op<"relu", [
    PredOpTrait<"x and y must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    /*QuantizableResult,*/
    SameOperandsAndResultShape]> {
  let summary = "Relu operator";

  let description = [{
    Element-wise Relu operator
      x -> max(0, x)
  }];

  let arguments = (ins CIR_TensorOf<[F32/*, QUI8, QI8, QI16*/]>:$x);

  let results = (outs CIR_TensorOf<[F32/*, QUI8, QI8, QI16*/]>:$y);

  // This builder doesn't work with quantized type, so it can only be used by
  // non-quantization tablegen patterns. Currently, it is used by the
  // elementwise-move reordering pattern in the optimize_patterns.td
  let builders = [
    OpBuilder<(ins "Value":$input),
    [{
      $_state.addOperands({input});
      $_state.addTypes(input.getType());
    }]>
  ];
}

def CIR_Relu6Op: CIR_Op<"relu6", [
    PredOpTrait<"x and y must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    /*QuantizableResult,*/
    SameOperandsAndResultShape]> {
  let summary = "Relu6 operator";

  let description = [{
    Element-wise Relu6 operator
      x -> max(0, min(6, x))
  }];

  let arguments = (ins CIR_TensorOf<[F32/*, QUI8, QI8*/]>:$x);

  let results = (outs CIR_TensorOf<[F32/*, QUI8, QI8*/]>:$y);

  // This builder doesn't work with quantized type, so it can only be used by
  // non-quantization tablegen patterns. Currently, it is used by the
  // elementwise-move reordering pattern in the optimize_patterns.td
  let builders = [
    OpBuilder<(ins "Value":$input),
    [{
      $_state.addOperands({input});
      $_state.addTypes(input.getType());
    }]>
  ];
}

def CIR_ReshapeOp: CIR_Op<"reshape", [
    /*QuantizableResult,*/
    Pure,
    /*SameOperandsAndResultsScale,*/
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Reshape operator";

  let description = [{
    Produces a tensor with the same values but different static shape defined
    by the output type.
  }];

  let arguments = (
    ins AnyTensor:$input,
    CIR_I32Tensor:$shape);

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 0b1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

def CIR_RsqrtOp: CIR_Op<"rsqrt", [Pure,
                                  //QuantizableResult,
                                  CIR_SameFirstOperandAndFirstResultElementType,
                                  SameOperandsAndResultShape]> {
  let summary = "Reciprocal of square root operator";

  let description = [{
    Computes element-wise reverse square root of input
  }];

  let arguments = (ins CIR_TensorOf<[F32/*, QI8, QI16*/]>:$x);

  let results = (outs CIR_TensorOf<[F32/*, QI8, QI16*/]>:$y);

  let hasFolder = 1;
}

def CIR_ShapeOp: CIR_Op<"shape", [
    /*QuantizableResult,*/
    Pure]> {
  let summary = "Shape operator";

  let description = [{
    Returns the shape of a tensor.
  }];

  let arguments = (ins AnyTensor:$input);

  let results = (outs CIR_TensorOf<[I32, I64]>:$output);

  DerivedTypeAttr out_type = DerivedTypeAttr<[{
    return getResult().getType().cast<TensorType>().getElementType();
  }]>;

  let hasOptions = 1;

  let hasFolder = 1;
}

def CIR_SqrtOp: CIR_Op<"sqrt", [
    Pure,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    // NOTE TFL_SqrtOp used TF_SameOperandsAndResultTypeResolveRef
    // REFER https://github.sec.samsung.net/one-project/circle-mlir/issues/292
    SameOperandsAndResultShape]> {
  let summary = "Square root operator";

  let description = [{
    Computes element-wise Square root of input
  }];

  let arguments = (ins CIR_FpTensor:$x);

  let results = (outs CIR_FpTensor:$y);

  let hasFolder = 1;

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

def CIR_SubOp : CIR_Op<"sub", [
    ResultsBroadcastableShape,
    BinaryOpSameElementTypeConstraint,
    CIR_RuntimePredOpTrait<"Operands do not have valid shapes",
      CPred<"Circle::VerifySubOpShapeConstraints(llvm::cast<SubOp>($_op))">>,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    Pure,
    /*QuantizableResult,*/
    ]> {
  let summary = "Subtraction operator";

  let description = [{
    Element-wise subtraction operation.
  }];

  let arguments = (
    ins CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, QI16*/]>:$lhs,
    CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, QI16*/]>:$rhs,
    CIR_AFAttr:$fused_activation_function);

  let results = (outs CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, QI16*/]>:$output);

  let hasFolder = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];

  let hasOptions = 1;
}

def CIR_SumOp: CIR_Op<"sum", [
    // TODO enable QuantizableResult,
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    // TODO(b/215655380): Re-enable this once there is 16-bit MLIR quantizer.
    // SameOperandsAndResultsScale,
    Pure]> {

  let summary = "Sum operator";

  let description = [{
    Computes the sum reduction along the specified axes
  }];

  let arguments = (ins
    CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, TFL_Quint8, QI16*/]>:$input,
    CIR_I32Tensor:$axes,
    BoolAttr:$keep_dims
  );

  let results = (outs CIR_TensorOf<[F32, I32, I64/*, QI8, QUI8, TFL_Quint8, QI16*/]>:$output);

  let hasOptions = 1;
  let customOption = "ReducerOptions";

  // TODO(b/215655380): Re-enable this once there is 16-bit MLIR quantizer.
  //
  //let extraClassDeclaration = [{
  //  // SameScalesOpInterface:
  //  bool RequiredSameOperandsAndResultsScale(bool sign, int bit_width) {
  //    // Eight-bit types don't require same operands and results scales.
  //    return bit_width != 8;
  //  }
  //}];
}

def CIR_TanhOp: CIR_Op<"tanh", [
    Pure,
    SameOperandsAndResultShape,
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    // TODO enable FixedOutputRangeInterface,
    // TODO enable QuantizableResult,
    ]> {
  let summary = "Hyperbolic tangent operator";

  let description = [{
    Computes element-wise Hyperbolic tangent of input
  }];

  let arguments = (ins CIR_TensorOf<[F32/*, QI8, QUI8, QI16, TFL_Quint8*/]>:$input);

  let results = (outs CIR_TensorOf<[F32/*, QI8, QUI8, QI16, TFL_Quint8*/]>:$output);

  // This builder doesn't work with quantized type, so it can only be used by
  // non-quantization tablegen patterns. Currently, it is used by the
  // elementwise-move reordering pattern in the optimize_patterns.td
  let builders = [
    OpBuilder<(ins "Value":$input),
    [{
      $_state.addOperands({input});
      $_state.addTypes(input.getType());
    }]>
  ];

  // TODO enable FixedOutputRangeInterface
  // let extraClassDeclaration = [{
  // // FixedOutputRangeInterface:
  // quant::UniformQuantizedType GetFixedOutputRange(
  //     bool is_signed, int bit_width) {
  //   auto result_type = getOutput().getType();
  //   // central_value = min_value / 2 + (max_value - 1) / 2 + 1
  //   // zero_point = central_value
  //   // scale = 1. / (central_value - min_value)
  //   return quant::GetFixedOutputRange(is_signed, bit_width, result_type,
  //       /*scale=*/1.0 / (1<<(bit_width-1)), /*zero_point=*/0);
  // }
  // }];
}

def CIR_TransposeOp : CIR_Op<"transpose", [
    Pure,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    CIR_OperandHasRankAtMost<0, 5>,
    CIR_OperandHasRank<1, 1>,
    PredOpTrait<"input and output must have same element type", CIR_TCresVTEtIsSameAsOp<0, 0>>/*,
    SameOperandsAndResultsScale*/]> {
  let summary = "Transpose operator";

  let description = [{
    Returns the Transpose of x
  }];

  let arguments = (ins
    CIR_TensorOf<[I32, F32, I8, UI8, /*QI8, QUI8, CIR_Quint8,*/ I1, I64/*, QI16*/]>:$input,
    CIR_TensorOf<[I32]>:$perm
  );

  let results = (outs
    CIR_TensorOf<[I32, F32, I8, UI8, /*QI8, QUI8, CIR_Quint8,*/ I1, I64/*, QI16*/]>:$output
  );

  let hasVerifier = 1;

  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$perm),
    [{ BuildTransposeOp(&$_builder, $_state, input, perm); }]>
  ];

  let extraClassDeclaration = [{
    // Quantized axes are verified in the Verify function.
    bool RequiredSameQuantizedAxes() { return false; }
  }];
}

#endif // CIRCLE_OPS
