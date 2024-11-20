/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>
#include <errno.h>
#include <sched.h>
#include <string.h>
#include <stdint.h>

#include "ggml-worker.h"

static TaskQueue *gQueue[NUM_CORES];
static pthread_t gThreads[NUM_CORES];
static pthread_mutex_t gGlobalMutex;

static void setSchedule(void)
{
  struct sched_param param;

  memset(&param, 0, sizeof(param));
  if (sched_getparam(0, &param) < 0) {
      printf("Failed to sched_getparam\n");
      return;
  }

  param.sched_priority = 1;
  if (sched_setscheduler(0, SCHED_RR, &param) == -1) {
      printf("error sched_setscheduler(SCHED_RR) errno:%d", errno);
      return;
  }
}

static void *worker(void *arg)
{
  WorkerArg *worker_arg = (WorkerArg *)arg;
  TaskQueue *queue = worker_arg->queue;
  int id = worker_arg->id;

  free(worker_arg);

  printf("sched_setaffinity : id[%d], cpu[%d] \n", id, sched_getcpu());

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

static TaskQueue *create_task_queue(int capacity)
{
  TaskQueue *queue = (TaskQueue *)malloc(sizeof(TaskQueue));
  queue->tasks = (Task *)malloc(sizeof(Task) * capacity);
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

void ggml_worker_init(void)
{ 

  setSchedule();

  for (int i = 0; i < NUM_CORES; i++)
  {
    gQueue[i] = create_task_queue(NUM_QUEUES);

    WorkerArg *worker_arg = (WorkerArg *)malloc(sizeof(WorkerArg));
    worker_arg->queue = gQueue[i];
    worker_arg->id = i + 1;
    pthread_create(&gThreads[i], NULL, worker, worker_arg);
  }
  pthread_mutex_init(&gGlobalMutex, NULL);
}

void ggml_worker_finalize(void)
{
  for (int i = 0; i < NUM_CORES; i++)
  {
    stop(gQueue[i]);
  }

  for (int i = 0; i < NUM_CORES; i++)
  {
    pthread_join(gThreads[i], NULL);
  }

  for (int i = 0; i < NUM_CORES; i++)
  {
    destroy_task_queue(gQueue[i]);
  }
}

void ggml_worker_submit(void (*func)(void *), void *arg)
{
  TaskQueue *queue;
  static int current_worker_id = 0;

  while(true)
  {
    current_worker_id = (current_worker_id + 1) % NUM_CORES;

    queue = gQueue[current_worker_id];

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
