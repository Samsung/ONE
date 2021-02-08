#ifndef __NNFW_TFLITE_COMPARATOR_INPUT_INITIALIZER_H__
#define __NNFW_TFLITE_COMPARATOR_INPUT_INITIALIZER_H__

#include "IOManager.h"

#include <misc/RandomGenerator.h>

namespace nnfw
{
namespace onert_cmp
{

class RandomInputInitializer
{
public:
  RandomInputInitializer(misc::RandomGenerator &randgen) : _randgen{randgen}
  {
    // DO NOTHING
  }

  void run(IOManager &manager);

private:
  template <typename T> void setValue(IOManager &manager, uint32_t tensor_idx);

private:
  nnfw::misc::RandomGenerator &_randgen;
};

class FileInputInitializer
{
public:
  FileInputInitializer(const std::vector<std::string> &files) : _files{files}
  {
    // DO NOTHING
  }

  void run(IOManager &manager);

private:
  const std::vector<std::string> &_files;
};

} // namespace onert_cmp
} // namespace nnfw

#endif // __NNFW_TFLITE_COMPARATOR_INPUT_INITIALIZER_H__
