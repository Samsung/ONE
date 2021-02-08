#include "tflite/OutputResetter.h"
#include "tflite/TensorView.h"

#include <misc/tensor/IndexIterator.h>

namespace nnfw
{
namespace tflite
{

void OutputResetter::run(::tflite::Interpreter &interp)
{
  for (const auto &tensor_idx : interp.outputs())
  {
    TfLiteTensor *tensor = interp.tensor(tensor_idx);
    if (tensor->type == kTfLiteInt32)
    {
      resetValue<int32_t>(interp, tensor_idx);
    }
    else if (tensor->type == kTfLiteUInt8)
    {
      resetValue<uint8_t>(interp, tensor_idx);
    }
    else if (tensor->type == kTfLiteInt8)
    {
      resetValue<int8_t>(interp, tensor_idx);
    }
    else if (tensor->type == kTfLiteBool)
    {
      resetValue<bool>(interp, tensor_idx);
    }
    else
    {
      assert(tensor->type == kTfLiteFloat32);

      resetValue<float>(interp, tensor_idx);
    }
  }
}

template <typename T> void OutputResetter::resetValue(::tflite::Interpreter &interp, int tensor_idx)
{
  auto tensor_view = nnfw::tflite::TensorView<T>::make(interp, tensor_idx);

  nnfw::misc::tensor::iterate(tensor_view.shape())
    << [&](const nnfw::misc::tensor::Index &ind) { tensor_view.at(ind) = 0; };
}

} // namespace tflite
} // namespace nnfw
