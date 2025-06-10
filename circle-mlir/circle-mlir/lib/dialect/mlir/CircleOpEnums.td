/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved
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

 // from tensorflow/compiler/mlir/lite/ir/tfl_op_enums.td

#ifndef CIRCLE_OP_ENUMS
#define CIRCLE_OP_ENUMS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

include "mlir/CircleOpInterfaces.td"

// A string attribute whose value are one of the values in `cases`.
// Referred TF_AnyStrAttrOf in tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td
class CIR_AnyStrAttrOf<list<string> cases> : StringBasedAttr<
  CPred<!foldl(
      "$_self.cast<StringAttr>().getValue() == \"" # !head(cases) # "\"",
      !foreach(case, !tail(cases),
               "$_self.cast<StringAttr>().getValue() == \"" # case # "\""),
      prev, cur, prev # " || " # cur)>,
  "string attribute whose value is " #
    !foldl(/*init*/!head(cases), /*list*/!tail(cases),
           prev, cur, prev # ", or " # cur)>;

// Allowed activation function cases
// These should match the ActivationFunctionType enum in Circle schema.
def CIR_AFEnum_None  : I32EnumAttrCase<"NONE", 0>;
def CIR_AFEnum_Relu  : I32EnumAttrCase<"RELU", 1>;
def CIR_AFEnum_Relu1 : I32EnumAttrCase<"RELU_N1_TO_1", 2>;
def CIR_AFEnum_Relu6 : I32EnumAttrCase<"RELU6", 3>;
def CIR_AFEnum_Tanh  : I32EnumAttrCase<"TANH", 4>;
def CIR_AFEnum_Sign  : I32EnumAttrCase<"SIGN_BIT", 5>;

def CIR_AFAttr : CIR_AnyStrAttrOf<[
      CIR_AFEnum_None.symbol,  CIR_AFEnum_Relu.symbol, CIR_AFEnum_Relu1.symbol,
      CIR_AFEnum_Relu6.symbol, CIR_AFEnum_Tanh.symbol, CIR_AFEnum_Sign.symbol
    ]>;

// Allowed padding cases
// These should match the padding enum in Circle schema.
def CIR_PADEnum_Same  : I32EnumAttrCase<"SAME", 0>;
def CIR_PADEnum_Valid : I32EnumAttrCase<"VALID", 1>;
def CIR_PaddingAttr : CIR_AnyStrAttrOf<[
      CIR_PADEnum_Same.symbol, CIR_PADEnum_Valid.symbol
    ]>;
def CIR_PAD_Same  : ConstantStrAttr<CIR_PaddingAttr, CIR_PADEnum_Same.symbol>;
def CIR_PAD_Valid : ConstantStrAttr<CIR_PaddingAttr, CIR_PADEnum_Valid.symbol>;

// MirrorPadding type attributes
def CIR_MIRRORPAD_Reflect : I32EnumAttrCase<"REFLECT", 0>;
def CIR_MIRRORPAD_Symmetric : I32EnumAttrCase<"SYMMETRIC", 1>;
def CIR_MirrorPaddingType : I32EnumAttr<"MirrorPaddingType", "mirror_pad_enum", [
      CIR_MIRRORPAD_Reflect, CIR_MIRRORPAD_Symmetric
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::Circle";
}
def CIR_MirrorPaddingAttr : EnumAttr<CIR_Dialect, CIR_MirrorPaddingType,
    "mirror_pad_attr">;

// FullyConnectedOptionsWeightFormat attributes
def CIR_FCWOEnum_Default         : I32EnumAttrCase<"DEFAULT", 0>;
def CIR_FCWOEnum_Shuffled4x16i8  : I32EnumAttrCase<"SHUFFLED4x16INT8", 1>;
def CIR_FullyConnectedOptionsWeightFormatAttr :
    CIR_AnyStrAttrOf<[
      CIR_FCWOEnum_Default.symbol,
      CIR_FCWOEnum_Shuffled4x16i8.symbol
    ]>;

def CIR_ConstBytesAttr : AttrDef<CIR_Dialect, "ConstBytes"> {
  let summary = "A string attribute representation of compiled bytes";
  let description = [{
    Syntax Examples:

    ```mlir
    #cir<const_bytes : "0xDEADBEEF">
    ```
  }];
  let mnemonic = "const_bytes";
  let parameters = (ins StringRefParameter<"">:$value);
  let hasCustomAssemblyFormat = 1;
}

#endif // CIRCLE_OP_ENUMS
