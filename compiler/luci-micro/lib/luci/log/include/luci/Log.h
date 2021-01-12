#ifndef LOG_H
#define LOG_H
#include <iostream>
// #include <ILogger.h>
// extern Teddy::ILogger &g_ilogger;
#define DEBUGLVL 
#define INFOLVL 

#define LOGGER(x) std::cout
#define INFO(x) std::cout
#define WARN(x) std::cout

#define VERBOSE(name, lv) std::cout
#define fmt(x) (const char *)(x)
#endif // LOG_H
