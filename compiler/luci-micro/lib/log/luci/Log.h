#ifndef LOG_H
#define LOG_H
#include <iostream>
// #include <ILogger.h>
// extern Teddy::ILogger &g_ilogger;
// #define DEBUGLVL (Teddy::TraceLevel_t::DEBUG)
// #define INFOLVL (Teddy::TraceLevel_t::INFO)

#define LOGGER(x) std::cout
#define INFO(x) std::cout
#define fmt(x) (const char *)(x)
#endif // LOG_H
