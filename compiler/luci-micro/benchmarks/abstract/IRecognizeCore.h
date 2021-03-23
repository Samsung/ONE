#ifndef IRECOGNIZECORE_H
#define IRECOGNIZECORE_H

#include <vector>

#include "ILogger.h"
namespace Teddy {
namespace AI {
class IRecognizeCore {
 public:
  virtual void process_frame(std::vector<int16_t> vec) = 0;
  bool is_ready = false;

 protected:
  ILogger *_log;
};
};      // namespace AI
};      // namespace Teddy
#endif  // IRECOGNIZECORE_H
