#include "tflite/OutputResetter.h"
#include "tflite/TensorView.h"

#include <misc/tensor/IndexIterator.h>

namespace nnfw
{
namespace tflite
{

void OutputResetter::run(TfLiteInterpreter *interp)
{
  auto num_outputs = TfLiteInterpreterGetOutputTensorCount(interp);
  for (int32_t output_idx = 0; output_idx < num_outputs; output_idx++)
  {
    auto tensor = TfLiteInterpreterGetInputTensor(interp, output_idx);
    if (tensor->type == kTfLiteInt32)
    {
      resetValue<int32_t>(interp, output_idx);
    }
    else if (tensor->type == kTfLiteUInt8)
    {
      resetValue<uint8_t>(interp, output_idx);
    }
    else if (tensor->type == kTfLiteInt8)
    {
      resetValue<int8_t>(interp, output_idx);
    }
    else if (tensor->type == kTfLiteBool)
    {
      resetValue<bool>(interp, output_idx);
    }
    else
    {
      assert(tensor->type == kTfLiteFloat32);

      resetValue<float>(interp, output_idx);
    }
  }
}

template <typename T> void OutputResetter::resetValue(TfLiteInterpreter *interp, int output_idx)
{
  auto tensor_view = nnfw::tflite::TensorView<T>::makeOutputView(interp, output_idx);

  nnfw::misc::tensor::iterate(tensor_view.shape())
    << [&](const nnfw::misc::tensor::Index &ind) { tensor_view.at(ind) = 0; };
}

} // namespace tflite
} // namespace nnfw
