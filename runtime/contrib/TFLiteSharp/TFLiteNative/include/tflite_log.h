/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _TFLITE_LOG_H_
#define _TFLITE_LOG_H_

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/

#define ERROR 1
#define WARNING 2
#define INFO 3
#define DEBUG 4

#ifdef __TIZEN__
#include <dlog/dlog.h>
#ifdef LOG_TAG
#undef LOG_TAG
#endif // LOG_TAG
#define LOG_TAG "TFLITE_NATIVE"

#define TFLITE_NATIVE_LOG(log_level, format, args...) \
  do                                                  \
  {                                                   \
    switch (log_level)                                \
    {                                                 \
      case ERROR:                                     \
        LOGE(format, ##args);                         \
      case WARNING:                                   \
        LOGE(format, ##args);                         \
      default:                                        \
        LOGI(format, ##args);                         \
    }                                                 \
  } while (0)
#else // __TIZEN__
#define LEVEL_TO_STR(level)           \
  (((level) == ERROR)     ? "ERROR"   \
   : ((level) == WARNING) ? "WARNING" \
   : ((level) == INFO)    ? "INFO"    \
   : ((level) == DEBUG)   ? "DEBUG"   \
                          : "DEFAULT")
#define TFLITE_NATIVE_LOG(log_level, format, args...)      \
  do                                                       \
  {                                                        \
    printf("%s: %s: ", LEVEL_TO_STR(log_level), __FILE__); \
    printf(format, ##args);                                \
    printf("\n");                                          \
  } while (0)
#endif // __TIZEN__

#ifdef __cplusplus
}
#endif /*__cplusplus*/

#endif /*_TFLITE_LOG_H*/
