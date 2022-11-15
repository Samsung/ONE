/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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
// base app for qemu
static volatile unsigned int *const UART_DR = (unsigned int *)0x40011004;

void uart_print(const char *s)
{
  while (*s != '\0')
  {
    *UART_DR = *s;
    s++;
  }
}

int main() { uart_print("Hello, World!\n"); }
