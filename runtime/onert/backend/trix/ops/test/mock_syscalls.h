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
