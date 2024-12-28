.data
txt: .asciz "Hello RISC-V\n"
.text
.global main
main:
    addi sp, sp, -32
    sd ra, 0(sp)
    
    la a0, txt
    call printf

    ld ra, 0(sp)
    addi sp, sp, 32
    ret