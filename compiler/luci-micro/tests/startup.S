.type start, %function

.word stack_top
.word start

.global start
start:
        ldr r1, =c_entry
        bx r1
