#ifndef ICORE_HPP
#define ICORE_HPP
#include <mbed.h>
namespace LuciMicro
{
class ICore
{
public:
  ICore(const osPriority_t thread_priority = osPriorityNormal, const char *name = "Unnamed thread");
  virtual ~ICore();
  void start(void);
  void stop();

protected:
  virtual void setup(void) = 0;
  virtual void loop(void) = 0;
  Thread _thread;
  bool _is_stopped;
  void set_priority(const osPriority_t thread_priority);
};
}; // namespace LuciMicro

#endif // ICORE_HPP
