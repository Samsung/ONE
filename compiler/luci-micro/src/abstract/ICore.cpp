#include "include/abstract/ICore.hpp"

using namespace LuciMicro;
using namespace mbed;
ICore::ICore(const osPriority_t thread_priority, const char *name) : _is_stopped(true)
{
  // _thread = new Thread(thread_priority, 2 * OS_STACK_SIZE, nullptr, name);
}
ICore::~ICore()
{
  _is_stopped = true;
  // delete _thread;
}
void ICore::start(void)
{
  if (_is_stopped)
  {
    _is_stopped = false;
    setup();
    _thread.start(callback(this, &ICore::loop));
  }
}
void ICore::stop(void) { _is_stopped = true; }
void ICore::set_priority(const osPriority_t thread_priority)
{
  _thread.set_priority(thread_priority);
}
