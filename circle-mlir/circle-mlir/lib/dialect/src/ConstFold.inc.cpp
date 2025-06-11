/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

// from tensorflow/compiler/mlir/lite/ir/tfl_ops.cc

// !!!!!
// WARNING Do not include this file other than 'CircleDialect.cpp'
// !!!!!

namespace mlir
{
namespace Circle
{

namespace
{

// Returns new shape with rank 'new_dims' with padded ones on the
// left if needed.
inline std::vector<int64_t> GetPaddedShape(ArrayRef<int64_t> old_shape, int new_dims)
{
  std::vector<int64_t> new_shape(new_dims, 1);
  std::copy_backward(old_shape.begin(), old_shape.end(), new_shape.end());
  return new_shape;
}

// Helper method that given and 'current_index' representing
// index in broadcasted tensor, get the index in the flat original tensor.
// 'shape' is the original shape with padding to match result shape.
int64_t GetElementIndex(const std::vector<int64_t> &shape,
                        const std::vector<int64_t> &current_index)
{
  int64_t ind = 0;
  int64_t mul = 1;
  for (int i = shape.size() - 1; i >= 0; --i)
  {
    ind += (current_index[i] % shape[i]) * mul;
    mul *= shape[i];
  }
  return ind;
}

// Helper method that increment index represented in 'current_index_ptr'
// in the shape of 'result_shape'.
void IncrementIndex(ArrayRef<int64_t> result_shape, std::vector<int64_t> *current_index_ptr)
{
  std::vector<int64_t> &current_index = *current_index_ptr;
  for (int i = result_shape.size() - 1; i >= 0; --i)
  {
    current_index[i]++;
    if (current_index[i] == result_shape[i])
    {
      current_index[i] = 0;
    }
    else
    {
      break;
    }
  }
}

/// Performs const folding `calculate` with broadcast behavior on the two
/// attributes `operand1` and `operand2` and returns the result if possible.
/// This function assumes the both operands are verified to have value
/// attributes of broadcastable types.
template <class AttrElementT, class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = llvm::function_ref<ElementValueT(ElementValueT, ElementValueT)>>
Attribute ConstFoldBinaryOpDenseDense(Type result_type, DenseElementsAttr lhs,
                                      DenseElementsAttr rhs, const CalculationT &calculate)
{
  auto type = mlir::dyn_cast_or_null<ShapedType>(
    OpTrait::util::getBroadcastedType(lhs.getType(), rhs.getType()));
  if (!type)
  {
    return {};
  }

  const bool rhs_is_splat = rhs.isSplat();
  const bool lhs_is_splat = lhs.isSplat();

  // If both of them are splat, compute and return.
  if (lhs_is_splat && rhs_is_splat)
  {
    auto element_result =
      AttrElementT::get(type.getElementType(), calculate(lhs.getSplatValue<ElementValueT>(),
                                                         rhs.getSplatValue<ElementValueT>()));
    if (!element_result)
      return {};

    return DenseElementsAttr::get(type, element_result);
  }

  auto num_elements = type.getNumElements();

  SmallVector<ElementValueT, 16> new_values;
  new_values.reserve(num_elements);
  const auto result_shape = type.getShape();
  std::vector<int64_t> current_index(type.getRank(), 0);
  // Create the new shape with ones padded to the left.
  const std::vector<int64_t> lhs_new_shape =
    GetPaddedShape(lhs.getType().getShape(), type.getRank());
  const std::vector<int64_t> rhs_new_shape =
    GetPaddedShape(rhs.getType().getShape(), type.getRank());

  auto lhs_old_values = lhs.getValues<ElementValueT>();
  auto rhs_old_values = rhs.getValues<ElementValueT>();

  // Add each pair of the corresponding values in the dense elements
  // attributes.
  for (int64_t i = 0; i < num_elements; ++i)
  {
    // current_index represents the index
    // in the N-dimension tensor. GetElementIndex returns
    // the index in the flat representation of the original tensor
    // to use.
    const int64_t lhs_index = lhs_is_splat ? 0 : GetElementIndex(lhs_new_shape, current_index);
    const int64_t rhs_index = rhs_is_splat ? 0 : GetElementIndex(rhs_new_shape, current_index);

    new_values.push_back(
      calculate(*(lhs_old_values.begin() + lhs_index), *(rhs_old_values.begin() + rhs_index)));
    IncrementIndex(result_shape, &current_index);
  }
  return DenseElementsAttr::get(type, ArrayRef<ElementValueT>(new_values));
}

/// Performs const folding `calculate` with broadcast behavior on the two
/// attributes `operand1` and `operand2` and returns the result if possible.
/// This function assumes the two operands are verified to have value
/// attributes of broadcastable types.
template <class AttrElementT, class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = llvm::function_ref<ElementValueT(ElementValueT, ElementValueT)>>
Attribute ConstFoldBinaryOp(Type result_type, Attribute operand1, Attribute operand2,
                            const CalculationT &calculate)
{
  if (mlir::dyn_cast_or_null<DenseElementsAttr>(operand1) &&
      mlir::dyn_cast_or_null<DenseElementsAttr>(operand2))
  {
    return ConstFoldBinaryOpDenseDense<AttrElementT, ElementValueT>(
      result_type, mlir::cast<DenseElementsAttr>(operand1), mlir::cast<DenseElementsAttr>(operand2),
      calculate);
  }

  // TODO: support other attribute kinds

  return {};
}

/// Performs const folding with broadcast behavior on the two attributes in
/// `operands` and returns the result if possible.
/// Depending on the given `resultType`, either `floatCalculate` or
/// `intCalculate` is chosen to conduct the calculate.
Attribute ConstFoldBinaryOp(Type result_type, ArrayRef<Attribute> operands,
                            llvm::function_ref<APFloat(APFloat, APFloat)> float_calculate,
                            llvm::function_ref<APInt(APInt, APInt)> int_calculate)
{
  // Note: All types are wrapped in tensor types in Circle. E.g., f32 is
  // represented as tensor<f32>. So we are only handling tensor types here.
  auto type = mlir::dyn_cast<ShapedType>(result_type);
  if (!type)
    return {};

  auto elemType = type.getElementType();

  if (mlir::isa<FloatType>(elemType))
    return ConstFoldBinaryOp<FloatAttr>(result_type, operands[0], operands[1], float_calculate);

  if (elemType.isSignlessInteger())
    return ConstFoldBinaryOp<IntegerAttr>(result_type, operands[0], operands[1], int_calculate);

  return {};
}

/// Performs const folding a attributes `operand` and returns the result if possible.
/// The function currently asserts that the `result_type` to be a f32 tensor type.
/// TODO: Extend this function to handle integral tensor for ops like "logical_not".
Attribute ConstFoldUnaryOp(Type result_type, Attribute operand,
                           llvm::function_ref<APFloat(APFloat)> calculate)
{
  assert(IsF32ShapedType(result_type));
  auto result_shape_type = mlir::cast<ShapedType>(result_type);

  if (!result_shape_type.hasStaticShape())
    return {};

  if (auto dense_elements = mlir::dyn_cast_or_null<DenseElementsAttr>(operand))
  {
    SmallVector<APFloat, 16> new_values;
    const int num_elements = result_shape_type.getNumElements();
    new_values.reserve(num_elements);

    for (const APFloat &old_value : dense_elements.getValues<APFloat>())
    {
      new_values.push_back(calculate(old_value));
    }

    return DenseElementsAttr::get(result_shape_type, new_values);
  }

  return {};
}

} // namespace

template <typename T> bool getAsConstant(mlir::Value &input, std::vector<T> &values)
{
  mlir::DenseElementsAttr dataAttr;

  // Check if input is constant
  if (!matchPattern(input, m_Constant(&dataAttr)))
    return false;

  if (auto constOp = dyn_cast<mlir::Circle::ConstOp>(input.getDefiningOp()))
    dataAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValueAttr());
  else
    return false;

  if (dataAttr == nullptr)
    return false;

  auto valueIt = dataAttr.getValues<llvm::APInt>().begin();
  auto valueEd = dataAttr.getValues<llvm::APInt>().end();
  for (; valueIt != valueEd; ++valueIt)
  {
    T value = static_cast<T>((*valueIt).getSExtValue());
    values.push_back(value);
  }
  return true;
}

bool getAsConstant(mlir::Value &input, std::vector<int64_t> &values)
{
  return getAsConstant<int64_t>(input, values);
}

bool getAsConstant(mlir::Value &input, std::vector<int32_t> &values)
{
  return getAsConstant<int32_t>(input, values);
}

bool getAsConstant(mlir::Value &input, std::vector<bool> &values)
{
  return getAsConstant<bool>(input, values);
}

} // namespace Circle
} // namespace mlir
