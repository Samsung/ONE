/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define NUM_CORES 4
#define NUM_QUEUES 16

typedef struct {
  void (*func)(void *);
  void *arg;
} Task;

typedef struct {
  Task *tasks;
  int head;
  int tail;
  int capacity;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  bool stop;
} TaskQueue;

typedef struct {
  TaskQueue *queue;
  int id;
} WorkerArg;


void ggml_worker_init(void);
void ggml_worker_finalize(void);
void ggml_worker_submit(void (*func)(void *), void *arg);
