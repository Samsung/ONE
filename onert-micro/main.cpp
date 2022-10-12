#include <luci_interpreter/Interpreter.h>
#include "circlemodel.h"
static volatile unsigned int *const UART_DR = (unsigned int *)0x40011004;
static void uart_print(const char *s)
{
  while (*s != '\0')
  {
    *UART_DR = *s;
    s++;
  }
}
int main()
{
  luci_interpreter::Interpreter interpreter(reinterpret_cast<const char*>(circle_model_raw));
  uart_print("Hello, World!\n");
  while (1)
    ;
  return 0;
}