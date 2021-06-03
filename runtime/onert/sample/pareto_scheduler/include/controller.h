/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file  controller.h
 * @brief This file describes API for pareto front based feedback controller
 */

#ifndef _PARETO_CONTROLLER_H
#define _PARETO_CONTROLLER_H

#include "session.h"
#include "tracer.h"

// Threshold margins are defined in this file
#define PERCENTAGE_THRESHOLD_LATENCY 80
#define PERCENTAGE_THRESHOLD_MEMORY 100
// This time is modeled to bypass transient effects immediately after reconfiguration
#define THRESHOLD_WAIT_TIME 2000 // value in milliseconds

enum transient_lock_e
{
  T_DISABLED,
  T_STANDBY,
  T_ENABLED_FOR_TIME,
  T_ENABLED_FOR_MEMORY
};
extern JsonWriter *json;
extern ParetoOptimizer *opt;

class ParetoScheduler
{
private:
  RunSession *_session;             // session object
  unsigned long _available_memory;  // currently available memory
  unsigned long _reference_memory;  // setpoint reference
  transient_lock_e _transient_lock; // Transient state to enable/disable control
  float _transient_wait_time;       // cumulative wait time after reconfiguration
public:
  /**
   * @brief     Construct a new ParetoScheduler object
   * @param[in] RunSession *session.  Pointer to an initialized RunSession object.
   */
  ParetoScheduler(RunSession *session);
  /**
   * @brief     Monitor session inference latency and reconfigure backends if the latency
   *            exceeds a threshold margin,  on account of GPU contention
   *            by a competing session.
   * @param[in] float exec_time. Measured execution time in milliseconds from the current inference
   * run.
   * @param[in] int inference_cnt. Inference count is used for periodic tracing of system memory for
   * Chrome Trace.
   */
  void latency_monitoring(float exec_time, int inference_cnt);
  /**
   * @brief     Monitor system memory and reconfigure backends when memory becomes available. The
   * context here is that the session was initially reconfigured for higher latency/lower RSS memory
   * on account of GPU contention by a competing session. When the competing session finishes, it
   * releases memory back to OS, which is then exploited to revert back to a all-GPU based backend
   * setting for high speed.
   */
  void memory_monitoring(void);
};

#endif
