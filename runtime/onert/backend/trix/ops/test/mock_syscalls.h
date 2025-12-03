/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _MOCK_SYSCALLS_H_
#define _MOCK_SYSCALLS_H_

#include <sys/mman.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <stdio.h>

int open(const char *, int, ...) { return 0; }
void *mmap(void *, size_t, int, int, int, off_t) { return (void *)0x1; }
int munmap(void *, size_t) { return 0; }
int close(int) { return 0; }
int ioctl(int, unsigned long, ...) { return 0; }
size_t fread(void *, size_t, size_t, FILE *) { return 1; }
int fseek(FILE *, long, int) { return 0; }

#endif
