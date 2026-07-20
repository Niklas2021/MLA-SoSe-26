  .file "custom_vadd.s"
  .section .text.custom_vadd,"ax",@progbits
  .globl custom_vadd
  .p2align 4
  .type custom_vadd,@function
custom_vadd:
// Computes C = A + B + B  (64 BF16 elements)
// Calling convention: p0 = ptr_in0 (A), p1 = ptr_in1 (B), p2 = ptr_out (C)
//
// Latenzannahmen (aus build/vadd.s und build/matmul_normal.s abgeleitet):
//   vlda.conv -> vadd.f: 4 cycles apart / 3 Zwischenzyklen (aus build/vadd.s)
//   vadd.f -> vadd.f  : Abstand 3 ueber den Akkumulator-Feedback-Pfad
//   vadd.f -> vst.conv: Latenz 6 / 5 Zwischenzyklen (aus build/vadd.s)
//   mova/movx : 1
//
// Slot-Schedule:
//   Cycle  1:  A: vlda.conv cml0[p0,#0]    X: movx r0,#60
//   Cycle  2:  A: vlda.conv cmh0[p0,#64]
//   Cycle  3:  A: vlda.conv cml1[p1,#0]
//   Cycle  4:  A: vlda.conv cmh1[p1,#64]
//   Cycle  5..7: nop  (Load-Latenz für letzten Load in Cycle 4)
//   Cycle  8:  V: vadd.f dm0,dm0,dm1,r0         // dm0 = A + B
//   Cycle  9:  nop                              // vadd.f-Latenz (1/2)
//   Cycle 10:  nop                              // vadd.f-Latenz (2/2)
//   Cycle 11:  V: vadd.f dm0,dm0,dm1,r0         // dm0 = (A+B) + B
//   Cycle 12:  nop                              // Zwischenzyklus 1/5
//   Cycle 13:  nop                              // Zwischenzyklus 2/5
//   Cycle 14:  X: ret lr                        // Zwischenzyklus 3/5
//   Cycle 15:  nop                              // Zwischenzyklus 4/5 / delay slot 5
//   Cycle 16:  nop                              // Zwischenzyklus 5/5 / delay slot 4
//   Cycle 17:  S: vst.conv cml0[p2,#0]          // delay slot 3
//   Cycle 18:  S: vst.conv cmh0[p2,#64]         // delay slot 2
//   Cycle 19:  nop                              // delay slot 1
// => 19 VLIW-Zyklen total.

  // Cycle 1: erster Load + sign-mask in r0 parallel (movx auf X-Slot)
  vlda.conv.fp32.bf16 cml0, [p0, #0];   movx r0, #60
  // Cycle 2
  vlda.conv.fp32.bf16 cmh0, [p0, #64]
  // Cycle 3
  vlda.conv.fp32.bf16 cml1, [p1, #0]
  // Cycle 4
  vlda.conv.fp32.bf16 cmh1, [p1, #64]
  // Cycles 5..7: warten auf Load-Latenz des letzten Loads
  nop
  nop
  nop
  // Cycle 8: dm0 = A + B
  vadd.f dm0, dm0, dm1, r0
  // Cycles 9..10: vadd.f-Latenz (3 cycles apart) ueberbruecken
  nop
  nop
  // Cycle 11: dm0 = (A+B)+B
  vadd.f dm0, dm0, dm1, r0
  // Cycles 12..13: warten auf vadd.f -> vst.conv Latenz
  nop
  nop
  // Cycle 14: ret so platzieren, dass die Stores in Delay Slots 3/2 fallen
  ret lr
  // 5 Delay-Slots nach ret:
  nop                                                       // Zwischenzyklus 4/5 / Delay Slot 5
  nop                                                       // Zwischenzyklus 5/5 / Delay Slot 4
  vst.conv.bf16.fp32 cml0, [p2, #0]                         // Delay Slot 3
  vst.conv.bf16.fp32 cmh0, [p2, #64]                        // Delay Slot 2
  nop                                                       // Delay Slot 1
.Lfunc_end0:
  .size custom_vadd, .Lfunc_end0-custom_vadd
