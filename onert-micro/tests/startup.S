.type start, %function

.word stack_top
.word start

.global start
start:
        ldr r1, =main
        bx r1
