#include "include/abstract/ILogger.hpp"
using namespace LuciMicro;
ILogger::ILogger() {}
ILogger::~ILogger() {}
TraceLevel_t ILogger::get_trace_level(void) { return _trace_level; }
void ILogger::set_trace_level(TraceLevel_t trace_lvl) { _trace_level = trace_lvl; }
