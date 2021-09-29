#ifndef ILOGGER_HPP
#define ILOGGER_HPP

#include <helpers/include/Helpers.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
namespace LuciMicro
{
enum TraceLevel_t
{
  NONE = 0,
  ERROR,
  WARNING,
  INFO,
  DEBUG,
  MIN = NONE,
  MAX = DEBUG
};
struct ILogger
{
  ILogger();
  typedef std::ostream &(*ManipFn)(std::ostream &);
  typedef std::ios_base &(*FlagsFn)(std::ios_base &);
  template <class T> // int, double, strings, etc
  ILogger &operator<<(const T &output)
  {
    LOCK_GUARD(this->_mutex);
    _stream << output;
    return *this;
  }

  ILogger &operator<<(ManipFn manip) /// endl, flush, setw, setfill, etc.
  {
    LOCK_GUARD(this->_mutex);
    manip(_stream);

    if (manip == static_cast<ManipFn>(std::flush) || manip == static_cast<ManipFn>(std::endl))
    {
      this->flush(m_logLevel, _stream.str());
      _stream.str(std::string());
      _stream.clear();
    }
    return *this;
  }

  ILogger &operator<<(FlagsFn manip) /// setiosflags, resetiosflags
  {
    LOCK_GUARD(this->_mutex);
    manip(_stream);
    return *this;
  }

  ILogger &operator()(TraceLevel_t e)
  {
    LOCK_GUARD(this->_mutex);
    m_logLevel = e;
    return *this;
  }
  virtual ~ILogger();
  virtual void printf(const char *format, ...) = 0;
  virtual void debug(const char *format, ...) = 0;
  virtual void warning(const char *format, ...) = 0;
  virtual void error(const char *format, ...) = 0;
  virtual void info(const char *format, ...) = 0;
  virtual bool is_available(void) const = 0;
  static TraceLevel_t get_trace_level(void);
  static void set_trace_level(TraceLevel_t);

protected:
  virtual void lock(void) { _mutex.lock(); };
  virtual void unlock(void) { _mutex.unlock(); };
  virtual void flush(const TraceLevel_t &logLevel, const std::string &log_string) = 0;
  rtos::Mutex _mutex;

private:
  static TraceLevel_t _trace_level;
  std::stringstream _stream;
  TraceLevel_t m_logLevel = TraceLevel_t::INFO;
};
};     // namespace LuciMicro
#endif // ILOGGER_HPP
