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

#include <csignal>
#include <iostream>
#include "common.h"
#include "controller.h"
#include "memory_stats.h"
#include "pareto_lookup.h"
#include "tracer.h"
#include "session.h"
#include <thread>

MessageQueue_t mq;
JsonWriter *json;
ParetoOptimizer *opt;

void signalHandler(int signum)
{
  std::cout << "Interrupt signal (" << signum << ") received." << std::endl;
  json->add_instance_record("Thread Exit");
  json->write_to_file();
  exit(signum);
}

void initialize_globals(std::string config_file, std::string dumpfile)
{
  json = new JsonWriter(dumpfile);
  opt = new ParetoOptimizer(config_file);
  opt->initialize_maps();
}

void runtime_thread(std::string model)
{
  json->add_timed_record("Runtime", "B");

  json->add_timed_record("session prepare", "B");

  // Prepare session.
  RunSession my_session(model);
  ParetoScheduler p_sched(&my_session);
  my_session.load_session();
  // Prepare output
  json->add_timed_record("session prepare", "E");
  std::cout << "model loaded" << std::endl;
  pthread_mutex_lock(&mq.msgq_mutex);
  mq.msg_queue.push("loaded");
  pthread_mutex_unlock(&mq.msgq_mutex);
  pthread_cond_signal(&mq.msgq_condition);

  int inference_cnt = 0;
  bool controls_enabled = true;
  float exec_time;
  unsigned long free_mem;
  unsigned long mem_diff;
  float cumulative_exec_time = 0;
  while (1)
  {
    pthread_mutex_lock(&mq.msgq_mutex);
    pthread_cond_wait(&mq.msgq_condition, &mq.msgq_mutex);
    pthread_mutex_unlock(&mq.msgq_mutex);
    std::cout << "thread received signal" << std::endl;

    if (!mq.msg_queue.empty())
    {
      std::string s = mq.msg_queue.front();
      mq.msg_queue.pop();
      json->add_timed_record("session sync", "E");
      if (s == "infer")
      {
        // Initialize inputs.
        json->add_timed_record("session initialize", "B");
        my_session.initialize_inputs();
        json->add_timed_record("session initialize", "E");
        // Do inference
        exec_time = my_session.run_inference();
        std::cout << "Inference iteration: " << inference_cnt++ << " done" << std::endl;
        exec_time /= 1000.0;

        // Run Latency Monitoring
        p_sched.latency_monitoring(exec_time, inference_cnt);

        // Run memory monitoring
        p_sched.memory_monitoring();

        json->add_timed_record("session sync", "B");
        mq.msg_queue.push("inferDone");
        usleep(500);
        pthread_cond_signal(&mq.msgq_condition);
      }
      else if (s == "exit")
      {
        break;
      }
      else
      {
        std::cout << "unknown message " << s << std::endl;
      }
    }
  }

  my_session.close();
  json->add_timed_record("Runtime", "E");
  json->write_and_close_file();
  std::cout << "nnpackage " << model << " runs successfully." << std::endl;
}

int main(const int argc, char **argv)
{
  srand(time(NULL));
  signal(SIGINT, signalHandler);
  signal(SIGKILL, signalHandler);
  signal(SIGTERM, signalHandler);
  initialize_globals(argv[2], argv[3]);

  std::thread runtime(runtime_thread, argv[1]);
  pthread_cond_wait(&mq.msgq_condition, &mq.msgq_mutex);
  ;
  mq.msg_queue.pop();
  pthread_mutex_unlock(&mq.msgq_mutex);
  auto n_iterations = std::stoi(argv[4]);
  for (auto i = 0; i < n_iterations; i++)
  {
    mq.msg_queue.push("infer");
    usleep(500);
    pthread_cond_signal(&mq.msgq_condition);
    pthread_mutex_lock(&mq.msgq_mutex);
    pthread_cond_wait(&mq.msgq_condition, &mq.msgq_mutex);
    pthread_mutex_unlock(&mq.msgq_mutex);
    mq.msg_queue.pop();
  }

  std::cout << "main calling runtime to exit.." << std::endl;
  std::string msg = "exit";
  pthread_mutex_lock(&mq.msgq_mutex);
  mq.msg_queue.push(msg);
  pthread_cond_signal(&mq.msgq_condition);
  pthread_mutex_unlock(&mq.msgq_mutex);
  std::cout << "main sent signal" << std::endl;
  runtime.join();
  std::cout << "main exiting" << std::endl;
  return 0;
}
