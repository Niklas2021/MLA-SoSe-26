  .file "zero.s"
  .section .text.zero,"ax",@progbits
  .globl zero
  .p2align 4
  .type zero,@function
zero:
  // Sets 512 bytes (one 2x2x8x8 bf16 out tile) to zero.
  // x0 is 512-bit (64 bytes) -> 8 stores cover the full tile.
  mov r0, #0
  vbcst.16 x0, r0
  nop                                 // vbcst latency 2 -> x0 ready for first vst
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  ret lr
  nop                                 // Delay Slot 5
  nop                                 // Delay Slot 4
  nop                                 // Delay Slot 3
  nop                                 // Delay Slot 2
  nop                                 // Delay Slot 1
.Lfunc_end0:
  .size zero, .Lfunc_end0-zero
