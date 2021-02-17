#ifndef __NNFW_TFLITE_COMPARATOR_IOMANAGER_H__
#define __NNFW_TFLITE_COMPARATOR_IOMANAGER_H__

#include "TensorView.h"

#include <nnfw.h>

#include <vector>
#include <map>

namespace nnfw
{
namespace onert_cmp
{

class IOManager
{

public:
  IOManager(nnfw_session *session);

public:
  void prepareIOBuffers();

  template <typename T> TensorView<T> inputView(uint32_t index);
  template <typename T> TensorView<T> outputView(uint32_t index);
  uint32_t inputs() const;
  uint32_t outputs() const;
  const nnfw_tensorinfo &inputTensorInfo(uint32_t index);
  const nnfw_tensorinfo &outputTensorInfo(uint32_t index);
  std::vector<uint8_t> &inputBase(uint32_t index);
  std::vector<uint8_t> &outputBase(uint32_t index);

private:
  void getInput(uint32_t index);
  void getOutput(uint32_t index);

private:
  nnfw_session *_session;
  std::vector<std::vector<uint8_t>> _inputs;
  std::vector<std::vector<uint8_t>> _outputs;
  std::vector<nnfw_tensorinfo> _input_infos;
  std::vector<nnfw_tensorinfo> _output_infos;
};

} // namespace onert_cmp
} // namespace nnfw

#endif // __NNFW_TFLITE_COMPARATOR_IOMANAGER_H__
