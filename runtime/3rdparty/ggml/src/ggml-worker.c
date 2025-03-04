/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <errno.h>
#include <sched.h>
#include <string.h>
#include <unistd.h>

#include "ggml.h"
#include "ggml-worker.h"

static TaskQueue *g_queue[MAX_NUM_OF_QUEUES];
static pthread_t g_threads[MAX_NUM_OF_THREADS];
static int g_worker_initialized = false;
static int g_num_of_threads;

static void setSchedule(void)
{
  struct sched_param param;

  memset(&param, 0, sizeof(param));
  if (sched_getparam(0, &param) < 0) {
      fprintf(stderr, "setSchedule: failed to sched_getparam errno:%d\n", errno);
      return;
  }

  param.sched_priority = 1;
  if (sched_setscheduler(0, SCHED_RR, &param) == -1) {
      fprintf(stderr, "setSchedule: failed to sched_setscheduler(SCHED_RR) errno:%d\n", errno);
      return;
  }
}

static TaskQueue *create_task_queue(int capacity)
{
  TaskQueue *queue = (TaskQueue *)malloc(sizeof(TaskQueue));
  if (!queue) {
    GGML_ABORT("Failed to allocate memory for queue");
    return NULL;
  }

  queue->tasks = (Task *)malloc(sizeof(Task) * capacity);
  if (!queue->tasks) {
    GGML_ABORT("Failed to allocate memory for tasks");
    free(queue);
    return NULL;
  }

  queue->head = 0;
  queue->tail = 0;
  queue->capacity = capacity;
  pthread_mutex_init(&queue->mutex, NULL);
  pthread_cond_init(&queue->cond, NULL);
  queue->stop = false;

  return queue;
}

static void destroy_task_queue(TaskQueue *queue)
{
  pthread_mutex_destroy(&queue->mutex);
  pthread_cond_destroy(&queue->cond);
  free(queue->tasks);
  free(queue);
}

static void stop(TaskQueue *queue)
{
  pthread_mutex_lock(&queue->mutex);
  queue->stop = true;
  pthread_cond_broadcast(&queue->cond);
  pthread_mutex_unlock(&queue->mutex);
}

static void *worker(void *arg)
{
  WorkerArg *worker_arg = (WorkerArg *)arg;
  TaskQueue *queue = worker_arg->queue;
  int id = worker_arg->id;

  free(worker_arg);

  while (true)
  {
    pthread_mutex_lock(&queue->mutex);

    while (queue->head == queue->tail && !queue->stop)
    {
      pthread_cond_wait(&queue->cond, &queue->mutex);
    }

    if (queue->stop && queue->head == queue->tail)
    {
      pthread_mutex_unlock(&queue->mutex);
      break;
    }

    Task task = queue->tasks[queue->tail];
    queue->tail = (queue->tail + 1) % queue->capacity;

    pthread_mutex_unlock(&queue->mutex);

    task.func(task.arg);
  }

  return NULL;
}

void ggml_worker_init(void)
{
  const long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);

  setSchedule(); // To improve latency, set scheduling policy to RR

  g_num_of_threads = cpu_count > MAX_NUM_OF_THREADS? MAX_NUM_OF_THREADS:cpu_count;

  for (int i = 0; i < g_num_of_threads; i++)
  {
    g_queue[i] = create_task_queue(MAX_NUM_OF_QUEUES);
    if (!g_queue[i]) {
      GGML_ABORT("Failed to create task queue");
      return;
    }    

    WorkerArg *worker_arg = (WorkerArg *)malloc(sizeof(WorkerArg));
    worker_arg->queue = g_queue[i];
    worker_arg->id = i + 1;
    if (pthread_create(&g_threads[i], NULL, worker, worker_arg) != 0) {
      GGML_ABORT("Failed to create worker thread");
      free(worker_arg);
      return;
    }
  }

  g_worker_initialized = true;
}

void ggml_worker_finalize(void)
{
  if(g_worker_initialized == false)
  {
    return;
  }

  for (int i = 0; i < g_num_of_threads; i++)
  {
    stop(g_queue[i]);
  }

  for (int i = 0; i < g_num_of_threads; i++)
  {
    pthread_join(g_threads[i], NULL);
    destroy_task_queue(g_queue[i]);
  }

  g_worker_initialized = false;
}

void ggml_worker_submit(void (*func)(void *), void *arg)
{
  TaskQueue *queue;
  static int current_worker_id = 0;
  static pthread_mutex_t submit_mutex = PTHREAD_MUTEX_INITIALIZER;

  while(true)
  {
    pthread_mutex_lock(&submit_mutex);
    current_worker_id = (current_worker_id + 1) % g_num_of_threads;
    pthread_mutex_unlock(&submit_mutex);

    queue = g_queue[current_worker_id];

    if(!queue->stop && ((queue->head + 1) % queue->capacity != queue->tail))
    {
      break;
    }
  }

  queue->tasks[queue->head].func = func;
  queue->tasks[queue->head].arg = arg;
  queue->head = (queue->head + 1) % queue->capacity;

  pthread_mutex_lock(&queue->mutex);
  pthread_cond_signal(&queue->cond);
  pthread_mutex_unlock(&queue->mutex);
}
