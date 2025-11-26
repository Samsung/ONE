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

#include "mock_syscalls.h"

int open(const char *pathname, int flags, ...)
{
  auto &manager = onert::backend::trix::ops::test::MockSyscallsManager::getInstance();

  // Handle variable arguments for open()
  if (flags & O_CREAT)
  {
    if (auto creatHook = manager.getOpenCreatHook())
    {
      va_list args;
      va_start(args, flags);
      mode_t mode = va_arg(args, mode_t);
      va_end(args);
      return creatHook(pathname, flags, mode);
    }
  }
  else
  {
    if (auto hook = manager.getOpenHook())
    {
      return hook(pathname, flags);
    }
  }
  return 0; // Default mock return value
}

void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
{
  auto &manager = onert::backend::trix::ops::test::MockSyscallsManager::getInstance();
  if (auto hook = manager.getMmapHook())
  {
    return hook(addr, length, prot, flags, fd, offset);
  }
  return (void *)0x1; // Default mock return value
}

int munmap(void *addr, size_t length)
{
  auto &manager = onert::backend::trix::ops::test::MockSyscallsManager::getInstance();
  if (auto hook = manager.getMunmapHook())
  {
    return hook(addr, length);
  }
  return 0; // Default mock return value
}

int close(int fd)
{
  auto &manager = onert::backend::trix::ops::test::MockSyscallsManager::getInstance();
  if (auto hook = manager.getCloseHook())
  {
    return hook(fd);
  }
  return 0; // Default mock return value
}

int ioctl(int fd, unsigned long request, ...)
{
  auto &manager = onert::backend::trix::ops::test::MockSyscallsManager::getInstance();
  if (auto hook = manager.getIoctlHook())
  {
    va_list args;
    va_start(args, request);
    void *arg = va_arg(args, void *);
    va_end(args);
    return hook(fd, request, arg);
  }
  return 0; // Default mock return value
}

FILE *fopen(const char *path, const char *mode)
{
  auto &manager = onert::backend::trix::ops::test::MockSyscallsManager::getInstance();
  if (auto hook = manager.getFopenHook())
  {
    return hook(path, mode);
  }
  return (FILE *)0x1; // Default mock return value
}

int fclose(FILE *stream)
{
  auto &manager = onert::backend::trix::ops::test::MockSyscallsManager::getInstance();
  if (auto hook = manager.getFcloseHook())
  {
    return hook(stream);
  }
  return 0;
}

size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
  auto &manager = onert::backend::trix::ops::test::MockSyscallsManager::getInstance();
  if (auto hook = manager.getFreadHook())
  {
    return hook(ptr, size, nmemb, stream);
  }
  return 1; // Default mock return value
}

int fseek(FILE *stream, long offset, int whence)
{
  auto &manager = onert::backend::trix::ops::test::MockSyscallsManager::getInstance();
  if (auto hook = manager.getFseekHook())
  {
    return hook(stream, offset, whence);
  }
  return 0; // Default mock return value
}
