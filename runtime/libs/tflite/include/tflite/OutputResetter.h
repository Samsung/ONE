#ifndef __NNFW_TFLITE_OUTPUT_RESETTER_H__
#define __NNFW_TFLITE_OUTPUT_RESETTER_H__

#include <tensorflow/lite/interpreter.h>

namespace nnfw
{
namespace tflite
{

class OutputResetter
{
public:
  OutputResetter()
  {
    // DO NOTHING
  }

  void run(::tflite::Interpreter &interp);

private:
  template <typename T> void resetValue(::tflite::Interpreter &interp, int tensor_idx);
};

} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_OUTPUT_RESETTER_H__
