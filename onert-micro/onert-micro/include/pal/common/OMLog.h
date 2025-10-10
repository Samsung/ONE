#ifndef ONERT_MICRO_OMLOG_H_
#define ONERT_MICRO_OMLOG_H_

#include <stdio.h>
#include <stdlib.h>

// Simple logging macro – can be replaced later with the project's logger
#define OM_LOG_ERROR(msg) \
  do { \
    fprintf(stderr, "[ERROR] %s:%d: %s\n", __FILE__, __LINE__, msg); \
  } while (0)

// Macro that logs a message and returns the given error code
#define OM_LOG_AND_RETURN(err, msg) \
  do { \
    OM_LOG_ERROR(msg); \
    return err; \
  } while (0)

#endif // ONERT_MICRO_OMLOG_H_
