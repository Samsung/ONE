SECTIONS
{
. = 0x0; /*ROM*/
.text : {
    startup.o(.text)
    test.o(.text)
    }
. = 0x20000000; /* RAM start*/
.data : { *(.data) }
.bss : { *(.bss) }
. = ALIGN(8);
. = . + 0x1000; /* stack size */

stack_top = .;
