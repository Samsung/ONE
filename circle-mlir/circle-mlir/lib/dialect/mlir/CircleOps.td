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
    Or<[CPred<"mlir::isa<UnrankedTensorType>($_op.getOperand(" # n # ").getType())">,
      CPred<"mlir::cast<ShapedType>($_op.getOperand(" # n #
        ").getType()).getRank() >= " # m>]>>;

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
  CPred<"mlir::isa<UnrankedTensorType>($_op.getOperand(" # n # ").getType())">;

// TODO: Some of these could be generalized and/or moved to more general
// location.
// Returns true if the n-th operand has unknown rank or has rank m.
class CIR_OperandHasRank<int n, int m> :
  PredOpTrait<"operand " # n # " is " # m # "-D",
    Or<[CIR_OperandIsUnrankedPred<n>,
      CPred<"mlir::cast<ShapedType>($_op.getOperand(" # n #
      ").getType()).getRank() == " # m>]>>;

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
    CPred<"mlir::cast<ShapedType>($_op.getOperand(" # n #
    ").getType()).getRank() <= " # m>]>;

// True if operand n is ranked and has a rank > dim.
class CIR_OperandIsRankedAndHasDimPred<int n, int dim> : And<[
  CPred<"mlir::isa<RankedTensorType>($_op.getOperand(" # n # ").getType())">,
  CPred<"mlir::cast<ShapedType>($_op.getOperand(" # n # ").getType()).getRank() > "
  # dim>]>;

// Returns true if the n-th operand is ranked and has a dimension length <=
// size at the rank dim.
class CIR_OperandDimIsAtMost<int n, int dim, int size> : And<[
  CIR_OperandIsRankedAndHasDimPred<n, dim>,
  CPred<"mlir::cast<ShapedType>($_op.getOperand(" # n # ").getType())"
      ".getShape()[" # dim # " ] <= " # size>]>;

class CIR_OperandRankEquals1DimOfOperand<int x, int y> :
  PredOpTrait<"operand " # x # "'s rank equals operand " # y # "'s size",
    Or<[CIR_OperandIsUnrankedPred<x>,
        CIR_OperandIsUnrankedPred<y>,
        CPred<"!mlir::cast<ShapedType>($_op.getOperand(" # y #
          ").getType()).hasStaticShape()">,
        CPred<"mlir::cast<ShapedType>($_op.getOperand(" # x #
          ").getType()).getRank() == "
          "mlir::cast<ShapedType>($_op.getOperand(" # y #
          ").getType()).getShape()[0]">]>>;

class CIR_OperandHasRankAtMost<int n, int m> :
  PredOpTrait<"operand " # n # " is at most " # m # "-D",
    CIR_OperandHasRankAtMostPred<n, m>>;

class CIR_OperandHasRankAtLeast<int n, int m> :
  PredOpTrait<"operand " # n # " is at least " # m # "-D",
    Or<[CIR_OperandIsUnrankedPred<n>,
      CPred<"mlir::cast<ShapedType>($_op.getOperand(" # n #
      ").getType()).getRank() >= " # m>]>>;

// Ensures the array attribute's size is within the given maximum size.
class CIR_ArrayMaxCount<int n> : AttrConstraint<
    CPred<"mlir::isa<ArrayAttr>($_self) && mlir::cast<ArrayAttr>($_self).size() <= " # n>,
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
    return mlir::cast<IntegerType>(mlir::cast<TensorType>(getResult().getType()).
            getElementType()).getWidth() > 32 ? circle::TensorType_INT64 :
            circle::TensorType_INT32;
    }], [{
      TypeAttr::get(mlir::cast<TensorType>(getResult().getType()).getElementType())
    }]>;
}

def CIR_AveragePool2DOp:
    CIR_Op<"average_pool_2d",
           [Pure,
            /*TODO enable SameOperandsAndResultsScale,*/
            /*TODO enable QuantizableResult,*/]> {
  let summary = "Average_pool_2d operator";

  let description = [{
    Performs average-pooling operation on input.
  }];

  let arguments = (
    ins CIR_TensorOf<[F32/*TODO enable, QI8, QUI8, QI16*/]>:$input,
    I32Attr:$filter_height,
    I32Attr:$filter_width,
    CIR_PaddingAttr:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w,
    CIR_AFAttr:$fused_activation_function
  );

  let results = (outs CIR_TensorOf<[F32/*TODO enable, QI8, QUI8, QI16*/]>:$output);

  let hasOptions = 1;
  let customOption = "Pool2DOptions";
}

def CIR_BatchMatMulOp : CIR_Op<"batch_matmul", [
   Pure,
   DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
   CIR_OperandHasAtleastRank<0, 2>,
   CIR_OperandHasAtleastRank<1, 2>,
   /* TODO enable QuantizableResult, */
   PredOpTrait<"x and output must have same element type or they are int8 and int32",
       Or<[CIR_TCresVTEtIsSameAsOp<0, 0>,
           And<[CPred<"getElementTypeOrSelf($_op.getOperand(0)).isInteger(8)">,
                CPred<"getElementTypeOrSelf($_op.getOperand(1)).isInteger(8)">,
                CPred<"getElementTypeOrSelf($_op.getResult(0)).isInteger(32)">]>]>>,
   CIR_RuntimePredOpTrait<"lhs and rhs of this op must have rank between [2, 5]",
     And<[CIR_OperandHasRankAtMostPred<0, 5>,
          CIR_OperandHasRankAtMostPred<1, 5>]>>,
   /* TODO enable DynamicRangeQuantizedOpInterface */]> {

  let summary = "Batch Matrix Multiply Operator";

  let description = [{
Performs a batched matrix multiplication on the inputs. Follows the
conventions of TensorFlow BatchMatMulV2, with support for unknown dimensions
in the batch dimensions and broadcasting.

    Inputs:
      `inputs[0]`: required: input LHS
      `inputs[1]`: required: input RHS
      `adjoint_lhs`: optional: Transpose LHS (default false)
      `adjoint_lhs`: optional: Transpose LHS (default false)
  }];

  // NOTE
  // The circle schema uses adjoint_lhs/adjoin_rhs, unlike the tensorflow schema.
  // Because the circle schema is based on TF v2.2-rc2 + more(`56d281c`) branch.
  // Please refer to the Revision Histroy comment in the top of circle_schema.fbs
  // file for detailed history.
  let arguments = (ins
    CIR_TensorOf<[F32/*, QI8, QI16*/, I8]>:$x,
    CIR_TensorOf<[F32/*, QI8, QI16*/, I8]>:$y,
    DefaultValuedAttr<BoolAttr, "false">:$adjoint_lhs,
    DefaultValuedAttr<BoolAttr, "false">:$adjoint_rhs,
    // Used in post-training dynamic range quantization. If the value is true,
    // input activations are asymmetrically quantized.
    OptionalAttr<BoolAttr>:$asymmetric_quantize_inputs
  );

   let results = (outs
    CIR_TensorOf<[F32/*, QI8, QI16*/, I32]>:$output
  );

  let hasOptions = 1;

  let extraClassDeclaration = [{
    // DynamicRangeQuantizedOpInterface:
    bool RequireAsymmetricQuantizeInputsAttr() { return true; }
    bool GetDynamicRangeQuantKernelSupport() { return true; }
    std::vector<int> GetQuantizableOperandIndices() { return {1}; }
  }];
}

def CIR_CastOp : CIR_Op<"cast", [
    Pure,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    SameOperandsAndResultShape]> {
  let summary = "Cast operator";

  let description = [{
    Casts input from input type to output type.
  }];

  let arguments = (ins
    CIR_TensorOf<[F32, I1, I16, UI16, I32, UI32, I64/*, TFL_Quint8*/, UI8, I8/*, Complex<F<32>>*/]>:$input
  );

  let results = (outs CIR_TensorOf<[F32, I1, I16, UI16, I32, UI32, I64/*, TFL_Quint8*/, UI8, I8/*, Complex<F<32>>*/]>:$output);

  // Circle's cast op does not utilize CastOptions, instead derives types
  // from the CircleTensors.
  let hasOptions = 0;

  let hasFolder = 1;
}

def CIR_ConcatenationOp : CIR_Op<"concatenation",
  [
    Pure,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    CIR_SameFirstOperandAndFirstResultElementType,
    // TODO enable SameOperandsAndResultsScale,
    // TODO enable QuantizableResult
  ]> {
  let summary = "Concatenation operator";

  let description = [{
    Concatenates tensors along one dimension
  }];

  let arguments = (
    ins CIR_VariadicTensorOf<
      [F32, I64, I32, I16, I8/*, QI8, QUI8*/, UI8, I1]>:$values,
    I32Attr:$axis,
    CIR_AFAttr:$fused_activation_function
  );

  let results = (outs
    CIR_TensorOf<
      [F32, I64, I32, I16, I8/*, QI8, QUI8*/, UI8, I1]>:$output
  );

  let hasOptions = 1;

  let hasFolder = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // SameScalesOpInterface:
    bool RequiredSameOperandsAndResultsScale(bool sign, int bit_width) {
      // uint8 doesn't require same operands and results scales.
      bool is_uint8 = !sign && (bit_width == 8);
      return !is_uint8;
    }
  }];
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

def CIR_DepthwiseConv2DOp :
    CIR_ConvOp<"depthwise_conv_2d", "Depthwise-separable convolution", /*3,*/
                [DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
                 /*DynamicRangeQuantizedOpInterface*/]> {
  let arguments = (
    ins CIR_TensorOf<[F32/*, QI8, QUI8, QI16*/]>:$input,
    CIR_TensorOf<[F32/*, QI8, QUI8*/]>:$filter,
    CIR_1DTensorOfOrNone<[F32, I32, I64]>:$bias,
    I32Attr:$dilation_h_factor,
    I32Attr:$dilation_w_factor,
    CIR_AFAttr:$fused_activation_function,
    CIR_PaddingAttr:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w,
    I32Attr:$depth_multiplier
  );

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    // AffineQuantizedOpInterface:
    int GetChannelDimIndex() { return 3; }
    int GetQuantizationDimIndex() { return 3; }
    // SparseOpInterface:
    std::vector<int> GetSparseOperands() { return {1}; }
    std::vector<std::vector<int>> GetFloatBlockSize() { return {}; }
    std::vector<std::vector<int>> GetQuantizedBlockSize() { return {}; }
    // DynamicRangeQuantizedOpInterface:
    bool GetDynamicRangeQuantKernelSupport() { return true; }
    std::vector<int> GetQuantizableOperandIndices() { return {1}; }
  }];
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

def CIR_EqualOp: CIR_Op<"equal", [
    Commutative,
    Pure,
    ResultsBroadcastableShape,
    CIR_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>,
    /*QuantizableResult,*/
    ComparisonOpSameElementTypeConstraint]> {
  let summary = "Equal operator";

  let description = [{
    Returns the truth element of x == y element-wise
  }];

  let arguments = (
    ins
    CIR_TensorOf<[I1, F32, I32, I64/*, QI8, QUI8, UI8, TFL_Str*/]>:$x,
    CIR_TensorOf<[I1, F32, I32, I64/*, QI8, QUI8, UI8, TFL_Str*/]>:$y
  );

  let results = (outs CIR_BoolTensor:$output);

  let hasFolder = 1;
}

def CIR_ExpOp: CIR_Op<"exp", [
    Pure,
    // NOTE TFL_ExpOp used TF_SameOperandsAndResultTypeResolveRef
    SameOperandsAndResultShape]> {
  let summary = "Natural exponentiation operator";

  let description = [{
    Performs element-wise natural exponentiation operation on input.
  }];

  let arguments = (ins CIR_FpTensor:$x);

  let results = (outs CIR_FpTensor:$y);

  let hasOptions = 0b1;

  // TODO enable TF::ArraysAreCastCompatible
  //let extraClassDeclaration = [{
  //  // Returns whether the return types are compatible.
  //  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  //    return TF::ArraysAreCastCompatible(l, r);
  //  }
  //}];
}

// From circle-mlir/externals/onnx-mlir/src/Dialect/ONNX/ONNXOps.td.inc
def CIR_ExpandOnnxOp : CIR_Op<"expand_onnx", [
    Pure]> {
  let summary = "Expand temporary ONNX Op";

  let description = [{
  Broadcast the input tensor following the given shape and the broadcast rule.
  The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
  Dimensions are right alignment;
  Two corresponding dimensions must have the same value, or one of them is equal to 1.
  Also, this operator is similar to numpy.broadcast_to(input, shape),
  but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
  It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
  or the shape.ndim < input.shape.ndim.
  }];
  let arguments = (ins
    CIR_TensorOf<[F32]>:$input,
    CIR_TensorOf<[I64]>:$shape);

  let results = (outs
    CIR_TensorOf<[F32]>:$output);

  let hasVerifier = 1;
  // ExpandOnnxOp with all constant inputs should convert to MulOp
  let hasCanonicalizer = 1;
}

def CIR_FloorOp: CIR_Op<"floor", [
    Pure,
    /*TF_SameOperandsAndResultTypeResolveRef*/]> {
  let summary = "Floor operator";

  let description = [{
    Returns element-wise floor value of the input.
  }];

  let arguments = (ins CIR_FpTensor:$x);

  let results = (outs CIR_FpTensor:$y);

  let hasFolder = 1;

  /*
  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return TF::ArraysAreCastCompatible(l, r);
    }
  }];
  */
}

def CIR_FloorModOp : CIR_Op<"floor_mod", [
    ResultsBroadcastableShape,
    Pure,
    BinaryOpSameElementTypeConstraint,
    PredOpTrait<"lhs and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    CIR_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>]> {
  let summary = "Division reminder";

  let description = [{
    Element-wise division reminder operation.
  }];

  let arguments = (
    ins CIR_TensorOf<[I8, I16, I32, I64, F32]>:$lhs,
    CIR_TensorOf<[I8, I16, I32, I64, F32]>:$rhs);

  let results = (outs CIR_TensorOf<[I8, I16, I32, I64, F32]>:$output);

  let builders = [CIR_BroadcastableBinaryBuilder];
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

def CIR_GatherOp : CIR_Op<"gather", [
    Pure,
    // TODO enable QuantizableResult,
    // TODO enable SameOperandsAndResultsScale,
    CIR_OperandHasAtleastRank<0, 1>,
    PredOpTrait<"params and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    // TODO enable DynamicRangeQuantizedOpInterface
  ]> {
  let summary = "Gather operator";

  let description = [{
    Gather slices from `params` axis `axis` according to `indices`.
  }];

  let arguments = (ins
    CIR_TensorOf<[F32/*, I1*/, I8, I16, I32, I64/*, TFL_Str, UI8, QI8, QUI8, QI16*/]>:$params,
    CIR_TensorOf<[I16, I32, I64]>:$indices,
    I32Attr:$axis,
    DefaultValuedOptionalAttr<I32Attr, "0">:$batch_dims
  );

  let results = (outs
    CIR_TensorOf<[F32/*, I1*/, I8, I16, I32, I64/*, TFL_Str, UI8, QI8, QUI8, QI16*/]>:$output
  );

  let hasOptions = 1;

  let hasVerifier = 1;

  let hasFolder = 1;

  let extraClassDeclaration = [{
    // DynamicRangeQuantizedOpInterface:
    std::vector<int> GetQuantizableOperandIndices() { return {0}; }
  }];
}

def CIR_GreaterOp : CIR_Op<"greater", [
    ResultsBroadcastableShape,
    ComparisonOpSameElementTypeConstraint,
    CIR_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>,
    /* QuantizableResult, */
    Pure]> {
  let summary = "Greater operator";

  let description = [{
    Element-wise greater operation.
  }];

  let arguments = (
    ins CIR_TensorOf<[F32, I32, I64/*, QUI8, QI8, TFL_Quint8*/]>:$lhs,
    CIR_TensorOf<[F32, I32, I64/*, QUI8, QI8, TFL_Quint8*/]>:$rhs);

  let results = (outs CIR_BoolTensor:$output);

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];
}

def CIR_HardSwishOp: CIR_Op<"hard_swish", [
    Pure,
    SameOperandsAndResultShape,
    /*QuantizableResult,*/
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>]> {
  let summary = "Hardswish activation function.";
  let description = [{
    Computes hard-swish activation function
      f(x) -> (x * relu6(x+3))/6
    element-wise.
  }];

  let arguments = (ins CIR_TensorOf<[F32/*, QUI8, QI8*/]>:$input);

  let results = (outs CIR_TensorOf<[F32/*, QUI8, QI8*/]>:$output);

  let hasOptions = 0;
}

// NOTE InstanceNormOp is added from scratch with looking around other ops
//      without deep understanding of TableDef, so there maybe something wrong.
//      we may have to fix errors when discovered.
def CIR_InstanceNormOp : CIR_Op<"instance_norm", [
    CIR_OperandHasRankAtMost<0, 4>,
    PredOpTrait<"input and output must have same element type", CIR_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>]> {
  let summary = "Instance normalization op";

  let description = [{
    Performs Instance normalization on input.

    Inputs:
      `inputs[0]`: required: the input tensor
  }];

  let arguments = (
    ins CIR_TensorOf<[F32]>:$input,
    CIR_1DTensorOf<[F32]>:$gamma,
    CIR_1DTensorOf<[F32]>:$beta,
    F32Attr:$epsilon,
    CIR_AFAttr:$fused_activation_function
  );
  let results = (outs CIR_TensorOf<[F32]>:$output);

  let hasOptions = 1;
  let customOption = "InstanceNormOptions";
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

def CIR_LogisticOp: CIR_Op<"logistic", [
    Pure,
    PredOpTrait<"x and y must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    SameOperandsAndResultShape,
    // TODO enable FixedOutputRangeInterface,
    // TODO enable QuantizableResult,
    ]> {
  let summary = "Logistic operator";

  let description = [{
    Computes element-wise Sigmoid of input
  }];

  let arguments = (ins CIR_TensorOf<[F32/*, QI8, QUI8, QI16, CIR_Quint8*/]>:$x);

  let results = (outs CIR_TensorOf<[F32/*, QI8, QUI8, QI16, CIR_Quint8*/]>:$y);

  // TODO enable FixedOutputRangeInterface
  // let extraClassDeclaration = [{
  // // FixedOutputRangeInterface:
  // quant::UniformQuantizedType GetFixedOutputRange(
  //     bool is_signed, int bit_width) {
  //   auto result_type = getY().getType();
  //   // zero_point = 0
  //   // scale = 1. / (max_value + 1)
  //   return quant::GetFixedOutputRange(is_signed, bit_width, result_type,
  //       /*scale=*/1.0 / (1<<(bit_width)),
  //       /*zero_point=*/-(1<<(bit_width-1)));
  // }
  // }];

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

def CIR_LogOp: CIR_Op<"log", [
    Pure,
    // NOTE TFL_LogOp used TF_SameOperandsAndResultTypeResolveRef
    SameOperandsAndResultShape]> {
  let summary = "Natural logarithm operator";

  let description = [{
    Performs element-wise natural logarithm operation on input.
  }];

  let arguments = (ins CIR_FpTensor:$x);

  let results = (outs CIR_FpTensor:$y);

  let hasFolder = 1;

  // Added for LogOfSoftmax
  let hasCanonicalizer = 1;

  // TODO enable TF::ArraysAreCastCompatible
  // let extraClassDeclaration = [{
  //   // Returns whether the return types are compatible.
  //   static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  //     return TF::ArraysAreCastCompatible(l, r);
  //   }
  // }];
}

def CIR_LogSoftmaxOp : CIR_Op<"log_softmax", [
    Pure,
    SameOperandsAndResultShape,
    PredOpTrait<"x and y must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    // TODO enable FixedOutputRangeInterface,
    // TODO enable QuantizableResult,
    ]> {
  let summary = "Log softmax operator";

  let description = [{
    Computes element-wise log softmax activations with the following formula

      input - log(reduce_sum(exp(input), dim))
  }];

  let arguments = (ins CIR_TensorOf<[F32/*, QUI8, QI8, TFL_Quint8*/]>:$input);

  let results = (outs CIR_TensorOf<[F32/*, QUI8, QI8, TFL_Quint8*/]>:$output);

  let hasOptions = 1;

  // TODO enable FixedOutputRangeInterface
  // let extraClassDeclaration = [{
  // // FixedOutputRangeInterface:
  // quant::UniformQuantizedType GetFixedOutputRange(
  //     bool is_signed, int bit_width) {
  //   auto result_type = getOutput().getType();
  //   // zero_point = max_value
  //   // scale = -log_softmax_output_min / (max_value + 1)
  //   return quant::GetFixedOutputRange(is_signed, bit_width, result_type,
  //       /*scale=*/16.0 / 256, /*zero_point=*/127);
  // }
  // }];
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

// TODO(ashwinm): Revisit the granularity of the PredOpTraits. We could
// break this into smaller PredOpTraits, each with more descriptive messages
// that would make it easier to trace failures OR, need a way to specify desc
// per Predicate inside the trait and get tablegen to use that to emit error
// message.
def MaxPoolOperandAndResultConstraints : PredOpTrait<"MaxPool2D operand and "
    "result types match specified constraints",
  And<[
    // The input and output tensors should have the same elemental type
    // and they should be one of the specified types below.
    TCopVTEtIs<0, AnyTypeOf<[F32/* TODO enable, QI8, QUI8*/]>>,
    CIR_TCresVTEtIsSameAsOp<0, 0>]>>;

def CIR_MaxPool2DOp : CIR_Op<"max_pool_2d", [
    CIR_OperandHasRank<0, 4>,
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    Pure,
    MaxPoolOperandAndResultConstraints,
    /*TODO enable SameOperandsAndResultsScale,*/
    /*TODO enable QuantizableResult,*/
    ]> {
  let summary = "Max Pool 2D op";

  let description = [{
    Performs max pool 2D on input.

    Inputs:
      `inputs[0]`: required: the input tensor
  }];

  let arguments = (
    ins CIR_TensorOf<[F32/*TODO enable, QUI8, QI8, QI16, TFL_Quint8*/]>:$input,
    CIR_PaddingAttr:$padding,
    I32Attr:$stride_w,
    I32Attr:$stride_h,
    I32Attr:$filter_width,
    I32Attr:$filter_height,
    CIR_AFAttr:$fused_activation_function
  );

  let results = (outs CIR_TensorOf<[F32/*TODO enable, QUI8, QI8, QI16, TFL_Quint8*/]>:$output);

  let hasOptions = 1;

  let customOption = "Pool2DOptions";
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

def CIR_MirrorPadOp: CIR_Op<"mirror_pad", [
                     /*SameOperandsAndResultsScale,*/
                     /*QuantizableResult,*/
                     DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
                     Pure/*,
                     CIR_OperandHasRank<1, 2>*/]> {
  let summary = "MirrorPad Operator. Pads a tensor with mirrored values.";

  let description = [{
    This operation pads a input with mirrored values according to the paddings
    you specify. paddings is an integer tensor with shape [n, 2],
    where n is the rank of input.
    For each dimension D of input, paddings[D, 0] indicates how many values
    to add before the contents of input in that dimension,
    and paddings[D, 1] indicates how many values to add after the contents of
    input in that dimension.

    Both paddings[D, 0] and paddings[D, 1] must be no greater than
    input.dim_size(D) (or input.dim_size(D) - 1)
    if copy_border is true (if false, respectively).

    The padded size of each dimension D of the output is:

    paddings(D, 0) + input.dim_size(D) + paddings(D, 1)
  }];

  let arguments = (ins
    CIR_TensorOf<[F32, I32, I64, I8, UI8/*, QI8, QUI8*/]>:$input,
    CIR_TensorOf<[I32, I64]>:$pad,
    CIR_MirrorPaddingAttr:$mode
  );

  let results = (outs
    CIR_TensorOf<[F32, I32, I64, I8, UI8/*, QI8, QUI8*/]>:$output
  );

  let hasOptions = 1;
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

def CIR_NegOp: CIR_Op<"neg", [
    Pure,
    // NOTE TFL_NegOp used TF_SameOperandsAndResultTypeResolveRef
    SameOperandsAndResultShape]> {
  let summary = "Negation operator";

  let description = [{
    Computes element-wise negation of input
  }];

  let arguments = (ins CIR_TensorOf<[F32, I32, I64]>:$x);

  let results = (outs CIR_TensorOf<[F32, I32, I64]>:$y);

  let hasOptions = 0b1;

  let hasFolder = 1;

  // NOTE ArraysAreCastCompatible requires too much fro TF
  // TODO enable ArraysAreCastCompatible
  //let extraClassDeclaration = [{
  //  // Returns whether the return types are compatible.
  //  static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  //    return ArraysAreCastCompatible(l, r);
  //  }
  //}];
}

def CIR_NotOp: CIR_Op<"logical_not", [
    Pure,
    SameOperandsAndResultShape]> {
  let summary = "Logical NOT operator";

  let description = [{
    Element-wise logical NOT operation.
  }];

  let arguments = (ins CIR_BoolTensor:$lhs);

  let results = (outs CIR_BoolTensor:$output);
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

def CIR_PadV2Op : CIR_Op<"padv2", [
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    //QuantizableResult,
    //SameOperandsAndResultsScale,
    CIR_OperandHasRankAtMost<0, 5>,
    CIR_OperandHasRank<1, 2>,
    CIR_OperandHasRank<2, 0>,
    CIR_OperandRankEquals1DimOfOperand<0, 1>,
    PredOpTrait<"the first dim size of the padding argument must be at most 5",
      Or<[CIR_OperandIsUnrankedPred<1>,
          CIR_OperandDimIsAtMost<1, 0, 5>]>>,
    PredOpTrait<"input and constant value operands must have same element type",
      CIR_TCopVTEtAreSameAt<0, 2>>]> {
  let summary = "Padding operator v2";

  let description = [{
    This operation pads a `input` according to the `paddings` and
    `constant_values` you specify. `paddings` is an integer tensor with shape
    `[Dn, 2]`, where n is the rank of `input`. For each dimension D of `input`,
    `paddings[D, 0]` indicates how many zeros to add before the contents of
    `input` in that dimension, and `paddings[D, 1]` indicates how many zeros to
    add after the contents of `input` in that dimension. `constant_values` is a
    scalar tensor of the same type as `input` that indicates the value to use
    for padding `input`.

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

  let arguments = (
    ins CIR_TensorOf<[F32, I32, I64, UI8/*, QI8, QUI8, TFL_Quint8*/]>:$input,
    CIR_I32OrI64Tensor:$padding,
    CIR_TensorOf<[F32, I32, I64, UI8/*, QI8, QUI8, TFL_Quint8*/]>:$constant_values);

  let results = (outs CIR_TensorOf<[F32, I32, I64, UI8/*, QI8, QUI8, TFL_Quint8*/]>:$output);

  let hasOptions = 1;

  let hasFolder = 1;
}

def CIR_PowOp : CIR_Op<"pow", [
    ResultsBroadcastableShape,
    Pure,
    CIR_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>]> {
  let summary = "Power operator";

  let description = [{
    Element-wise power operation.
  }];

  let arguments = (
    ins CIR_TensorOf<[F32, I32]>:$lhs,
    CIR_TensorOf<[F32, I32]>:$rhs);

  let results = (outs CIR_TensorOf<[F32, I32]>:$output);

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];

  let builders = [CIR_BroadcastableBinaryBuilder];
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
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
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

def CIR_ResizeBilinearOp: CIR_Op<"resize_bilinear", [
    Pure,
    /*QuantizableResult,*/
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    CIR_OperandHasRank<0, 4>,
    CIR_OperandHasRank<1, 1>,
    /*SameOperandsAndResultsScale*/]> {
  let summary = "ResizeBilinear Op";

  let description = [{
    Resize `images` to `size` using bilinear interpolation.
  }];

  let arguments = (ins
    CIR_TensorOf<[F32/*, TFL_Quint8, QUI8, QI8, QI16*/]>:$input,
    CIR_I32Tensor:$size,
    BoolAttr:$align_corners,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$half_pixel_centers
  );

  let results = (outs
    CIR_TensorOf<[F32/*, TFL_Quint8, QUI8, QI8, QI16*/]>:$output
  );

  let hasOptions = 1;
}

def CIR_ResizeNearestNeighborOp : CIR_Op<"resize_nearest_neighbor", [
    Pure,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    /*QuantizableResult,*/
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    CIR_OperandHasRank<0, 4>,
    CIR_OperandHasRank<1, 1>,
    /*SameOperandsAndResultsScale*/]> {
  let summary = "ResizeNearestNeighbor Op";

  let description = [{
    Resize `images` to `size` using nearest neighbor interpolation.
  }];

  let arguments = (ins
    CIR_TensorOf<[F32/*, TFL_Quint8, QUI8, QI8, QI16*/]>:$input,
    CIR_I32Tensor:$size,
    BoolAttr:$align_corners,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$half_pixel_centers
  );

  let results = (outs
    CIR_TensorOf<[F32/*, TFL_Quint8, QUI8, QI8, QI16*/]>:$output
  );

  let hasOptions = 1;
}

// NOTE this is temporary Op to hold scales for unknown input shape
def CIR_ResizeOnnxOp : CIR_Op<"resize_onnx", [
    Pure]> {
  let summary = "Resize temporary ONNX Op";

  let description = [{
    Resize temporary Op for unknown shape.
  }];

  let arguments = (ins
    CIR_TensorOf<[F32]>:$input,
    CIR_TensorOf<[F32]>:$scales,
    StrAttr:$mode
  );

  let results = (outs
    CIR_TensorOf<[F32]>:$output
  );

  let hasCanonicalizer = 1;
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

// Select has many instances in models where one or more of its operands
// are unranked. Therefore, we skip adding shape constraints here.
def CIR_SelectOp : CIR_Op<"select", [
  Pure,
  /*SameOperandsAndResultsScale,*/
  /*QuantizableResult,*/
  PredOpTrait<"operands have same element type", CIR_TCopVTEtAreSameAt<1, 2>>,
  PredOpTrait<"operands and result have same element type", CIR_TCresVTEtIsSameAsOp<0, 1>>]> {
  let summary = "Select operator";

  let description = [{
    Select values of 'x' if the corresponding value of 'condition' is true or
    the value of 'y' if false. There are valid condition input sizes:

    1. Either the same shape (in which case the select is elementwise), or
    2. condition must be Rank 1 and match over the first dimension.
  }];

  let arguments = (ins
    CIR_BoolTensor:$condition,
    CIR_TensorOf<[F32, I1, I8, I16, I32, I64/*, QI8, QUI8, QI16, TFL_Quint8*/]>:$x,
    CIR_TensorOf<[F32, I1, I8, I16, I32, I64/*, QI8, QUI8, QI16, TFL_Quint8*/]>:$y);

  let results = (outs
    CIR_TensorOf<[F32, I1, I8, I16, I32, I64/*, QI8, QUI8, QI16, TFL_Quint8*/]>:$output);

  // TODO(jpienaar): autogenerate this.
  let builders = [
    OpBuilder<(ins "Value":$condition, "Value":$x, "Value":$y),
    [{
    auto resultType = x.getType();
    $_state.addOperands({condition, x, y});
    $_state.types.push_back(resultType);
  }]>];

  let hasOptions = 1;

  let hasFolder = 1;
}

def CIR_SelectV2Op : CIR_Op<"select_v2", [
    ResultsBroadcastableShape,
    Pure,
    /*QuantizableResult,*/
    /*SameOperandsAndResultsScale,*/
    CIR_OperandsHaveSameShapesOrBroadcastableShape<[0, 1, 2], 5>,
    PredOpTrait<"operands have same element type", CIR_TCopVTEtAreSameAt<1, 2>>,
    PredOpTrait<"operands and result have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 1>>]> {
  let summary = "SelectV2 operator";

  let description = [{
    Select values of 'x' if the corresponding value of 'condition' is true or
    the value of 'y' if false. There are valid condition input sizes:

    1. Either the same shape (in which case the select is elementwise), or
    2. Broadcastable shapes between 'condition', 'x' and 'y'.
  }];

  let arguments = (ins
    CIR_BoolTensor:$condition,
    CIR_TensorOf<[F32, I1, I8, I16, I32, I64, UI32/*, QI8, QUI8, QI16, TFL_Quint8*/]>:$x,
    CIR_TensorOf<[F32, I1, I8, I16, I32, I64, UI32/*, QI8, QUI8, QI16, TFL_Quint8*/]>:$y);

  let results = (outs
    CIR_TensorOf<[F32, I1, I8, I16, I32, I64, UI32/*, QI8, QUI8, QI16, TFL_Quint8*/]>:$output);

  let hasOptions = 1;

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

def CIR_SinOp: CIR_Op<"sin", [
    Pure,
    /*TF_SameOperandsAndResultTypeResolveRef*/
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    SameOperandsAndResultShape]> {
  let summary = "Sine operator";

  let description = [{
    Computes element-wise Sine of input
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

def CIR_SliceOp : CIR_Op<"slice", [
    /*QuantizableResult,*/
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    /*SameOperandsAndResultsScale,*/
    CIR_OperandHasRankAtMost<0, 5>,
    CIR_OperandHasRankAtMost<1, 1>,
    CIR_OperandHasRankAtMost<2, 1>]> {
  let summary = "Return a slice from 'input'.";

  let description = [{
The output tensor is a tensor with dimensions described by 'size'
whose values are extracted from 'input' starting at the offsets in
'begin'.

`begin` is zero-based; `size` is one-based. If size[i] is -1, all remaining
elements in dimension i are included in the slice. In other words, this is
equivalent to setting:
  size[i] = input.dim_size(i) - begin[i]

*Requirements*:
  0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)
  }];

  let arguments = (ins
    CIR_TensorOf<[F32, I32, I64, I8, UI8/*, I1, CIR_Str, QI8, QUI8, TFL_Quint8, QI16*/]>:$input,
    CIR_I32OrI64Tensor:$begin,
    CIR_I32OrI64Tensor:$size
  );

  let results = (outs
    CIR_TensorOf<[F32, I32, I64, I8, UI8/*, I1, CIR_Str, QI8, QUI8, TFL_Quint8, QI16*/]>:$output
  );

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def CIR_SoftmaxOp : CIR_Op<"softmax", [
    Pure,
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    CIR_OperandHasRankAtLeast<0, 1>,
    SameOperandsAndResultShape,
    // TODO enable QuantizableResult,
    // TODO enable FixedOutputRangeInterface,
    ]> {
  let summary = "Softmax operator";

  let description = [{
    Computes element-wise softmax activations with the following formula

      exp(input) / tf.reduce_sum(exp(input * beta), dim)
  }];

  let arguments = (
    ins CIR_TensorOf<[F32/*, QI8, QUI8, TFL_Quint8, QI16*/]>:$input,
    F32Attr:$beta
  );

  let results = (outs CIR_TensorOf<[F32/*, QI8, QUI8, TFL_Quint8, QI16*/]>:$output);

  let hasOptions = 1;

  // TODO enable FixedOutputRangeInterface
  // let extraClassDeclaration = [{
  // // FixedOutputRangeInterface:
  // quant::UniformQuantizedType GetFixedOutputRange(
  //     bool is_signed, int bit_width) {
  //   if (bit_width != 8 && bit_width != 16) { return nullptr; }
  //   auto result_type = getOutput().getType();
  //   // zero_point = 0
  //   // scale = 1. / (max_value + 1)
  //   return quant::GetFixedOutputRange(is_signed, bit_width, result_type,
  //       /*scale=*/1.0 / (bit_width == 8 ? (1<<(bit_width)) : (1<<(bit_width-1))),
  //       /*zero_point=*/bit_width == 8 ? -(1<<(bit_width-1)): 0);
  // }
  // }];
}

def CIR_SplitVOp : CIR_Op<"split_v", [
    Pure,
    /*QuantizableResult,*/
    /*SameOperandsAndResultsScale*/]> {
  let summary = "Splits a tensor into `num_split` tensors along one dimension.";

  let description = [{
    Splits the `value` tensor along `split_dim` into a number of sub-tensors
    with same shape as the original one, except for `split_dim`. The grouping
    of the resultant sub-tensors is decided by `size-splits`. Same as tf.SplitV.
  }];

  let arguments = (ins
    CIR_TensorOf<[F32, I16, I32, I64, I8, UI8/*, QI8, QUI8, QI16*/]>:$value,
    CIR_1DTensorOf<[I32], [I32]>:$size_splits,
    CIR_0DTensorOf<[I32], [I32]>:$split_dim,
    ConfinedAttr<I32Attr, [IntPositive]>:$num_splits
  );

  let results = (outs
    CIR_VariadicTensorOf<[F32, I16, I32, I64, I8, UI8/*, QI8, QUI8, QI16*/]>:$outputs
  );

  let hasVerifier = 1;

  let hasOptions = 1;
}

def CIR_SqrtOp: CIR_Op<"sqrt", [
    Pure,
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    // NOTE TFL_SqrtOp used TF_SameOperandsAndResultTypeResolveRef
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

def CIR_SqueezeOp: CIR_Op<"squeeze", [Pure,
                                      /*QuantizableResult,*/
                                      /*SameOperandsAndResultsScale*/]> {
  let summary = "Removes dimensions of size 1 from the shape of a tensor.";

  let description = [{
Given a tensor `input`, this operation returns a tensor of the same type with
all dimensions of size 1 removed. If you don't want to remove all size 1
dimensions, you can remove specific size 1 dimensions by specifying
`squeeze_dims`.

For example:

```
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t)) ==> [2, 3]
```

Or, to remove specific size 1 dimensions:

```
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
```
  }];

  let arguments = (ins
    AnyTensor:$input,
    ConfinedAttr<DefaultValuedOptionalAttr<I64ArrayAttr, "{}">, [CIR_ArrayMaxCount<8>]>:$squeeze_dims
  );

  let results = (outs
    AnyTensor:$output
  );

  let hasFolder = 1;
  let hasOptions = 1;

  let customOption = "SqueezeOptions";
}

def CIR_StridedSliceOp: CIR_Op<"strided_slice", [
    Pure,
    /*QuantizableResult,*/
    DeclareOpInterfaceMethods<CIR_ShapeInferenceOpInterface>,
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>,
    /*SameOperandsAndResultsScale,*/
    CIR_OperandHasRankAtMost<0, 5>,
    CIR_OperandHasRank<1, 1>,
    CIR_OperandHasRank<2, 1>,
    CIR_OperandHasRank<3, 1>
  ]> {
  let summary = "StridedSlice Op";

  let description = [{
    Return a strided slice from `input`.
  }];

  let arguments = (ins
    CIR_TensorOf<[F32, I32, I64, I8/*, UI8, QI8, QUI8, I1*/, I16/*, QI16, TFL_Quint8, TFL_Str*/]>:$input,
    CIR_I32Tensor:$begin,
    CIR_I32Tensor:$end,
    CIR_I32Tensor:$strides,

    I32Attr:$begin_mask,
    I32Attr:$end_mask,
    I32Attr:$ellipsis_mask,
    I32Attr:$new_axis_mask,
    I32Attr:$shrink_axis_mask
  );

  let results = (outs
    CIR_TensorOf<[F32, I32, I64, I8/*, UI8, QI8, QUI8, I1*/, I16/*, QI16, TFL_Quint8, TFL_Str*/]>:$output
  );

  // TFLite kernel only supports up to 5D input including added axis.
  let hasVerifier = 1;

  let hasOptions = 1;

  let hasFolder = 1;

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    // Folding 1-dim of input
    OpFoldResult foldOneDimension();
    // Folding reverse of input when stride is -1
    OpFoldResult foldReverseInput();
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

def CIR_TileOp: CIR_Op<"tile", [
    Pure,
    /*SameOperandsAndResultsScale,*/
    /*QuantizableResult,*/
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 0>>]> {
  let summary = "Tile operator.";
  let description = [{
    Constructs a tensor by tiling a given tensor.

   This operation creates a new tensor by replicating input
   multiples times. The output tensor's i'th dimension has
   input.dims(i) * multiples[i] elements, and the values of input
   are replicated multiples[i] times along the 'i'th dimension.
   For example, tiling [a b c d] by [2] produces [a b c d a b c d].
  }];

  let arguments = (ins
    CIR_TensorOf<[F32, I1, I32, I64/*, UI8, QI8, QUI8, TFL_Str*/]>:$input,
    CIR_I32OrI64Tensor:$multiples);

  let results = (outs
    CIR_TensorOf<[F32, I1, I32, I64/*, UI8, QI8, QUI8, TFL_Str*/]>:$output);

  let hasOptions = 0;
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

def CIR_TransposeConvOp: CIR_Op<"transpose_conv", [
    Pure,
    CIR_OperandHasRank<0, 1>,
    CIR_OperandHasRank<1, 4>,
    CIR_OperandHasRank<2, 4>,
    PredOpTrait<"input and output must have same element type",
      CIR_TCresVTEtIsSameAsOp<0, 2>>,
    /*AccumulatorUniformScale<3, 1, 2>,*/
    /*AffineQuantizedOpInterface, AffineOpCoefficient<0, 1>,*/
    /*QuantizableResult,*/
    CIR_SparseOp/*,
    DynamicRangeQuantizedOpInterface*/]> {
  let summary = "Transpose convolution operator";

  let description = [{
    Performs transpose convolution operation on input.
  }];

  let arguments = (ins
    CIR_I32Tensor:$output_shape,
    CIR_TensorOf<[F32/*, QI8, QUI8, QI16*/]>:$weights,
    CIR_TensorOf<[F32/*, QI8, QUI8, QI16*/]>:$input,
    CIR_TensorOfOrNone<[F32/*, QI32, I64*/]>:$bias,
    CIR_PaddingAttr:$padding,
    ConfinedAttr<I32Attr, [IntPositive]>:$stride_h,
    ConfinedAttr<I32Attr, [IntPositive]>:$stride_w
  );

  let results = (outs CIR_TensorOf<[F32/*, QI8, QUI8, QI16*/]>:$output);

  let hasOptions = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // AffineQuantizedOpInterface:
    int GetChannelDimIndex() { return 0; }
    int GetQuantizationDimIndex() { return 0; }
    // SparseOpInterface:
    std::vector<int> GetSparseOperands() { return {1}; }
    std::vector<std::vector<int>> GetFloatBlockSize() { return {}; }
    std::vector<std::vector<int>> GetQuantizedBlockSize() { return {}; }
    // DynamicRangeQuantizedOpInterface:
    std::vector<int> GetQuantizableOperandIndices() { return {1}; }
  }];
}

// From submodules/onnx-mlir/src/Dialect/ONNX/ONNXOps.td.inc
def CIR_UnsqueezeOnnxOp:CIR_Op<"unsqueeze_onnx", [
    Pure]> {
  let summary = "Unsqueeze temporary ONNX Op";

  let description = [{
    Insert single-dimensional entries to the shape of an input tensor (`data`).
    Takes one required input `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).

    For example, given an input tensor (`data`) of shape [3, 4, 5], then
    Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing same data as `data` but with shape [1, 3, 4, 5, 1].

    The input `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
    The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
    Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1].
    The order of values in `axes` does not matter and can come in any order.
  }];

  let arguments = (ins
    CIR_TensorOf<[F32]>:$data,
    CIR_TensorOf<[I64]>:$axes);

  let results = (outs
    CIR_TensorOf<[F32]>:$expanded);

  // UnsqueezeOnnxOp should be converted to ResizeOp
  let hasCanonicalizer = 1;
}

#endif // CIRCLE_OPS
