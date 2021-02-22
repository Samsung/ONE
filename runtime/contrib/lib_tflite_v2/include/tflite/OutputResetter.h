#ifndef __NNFW_TFLITE_OUTPUT_RESETTER_H__
#define __NNFW_TFLITE_OUTPUT_RESETTER_H__

#include <tensorflow/lite/c/c_api.h>

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

  void run(TfLiteInterpreter *interp);

private:
  template <typename T> void resetValue(TfLiteInterpreter *interp, int output_idx);
};

} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_OUTPUT_RESETTER_H__
