  .file "custom_vadd.s"
  .section .text.custom_vadd,"ax",@progbits
  .globl custom_vadd
  .p2align 4
  .type custom_vadd,@function
custom_vadd:
// Computes C = A + B + B  (64 BF16 elements)
// Calling convention: p0 = ptr_in0 (A), p1 = ptr_in1 (B), p2 = ptr_out (C)
//
// Slot-Schedule (Annahmen: vlda.conv-Latenz=5, vadd.f-Latenz=2, mova-Latenz=1):
//   Cycle  1:  A: vlda.conv cml0[p0,#0]    X: movx r0,#60
//   Cycle  2:  A: vlda.conv cmh0[p0,#64]
//   Cycle  3:  A: vlda.conv cml1[p1,#0]
//   Cycle  4:  A: vlda.conv cmh1[p1,#64]
//   Cycle  5..8: nop  (Load-Latenz für letzten Load in Zyklus 4)
//   Cycle  9:  V: vadd.f dm0,dm0,dm1,r0        // dm0 = A + B
//   Cycle 10:  nop                              // vadd.f-Latenz
//   Cycle 11:  V: vadd.f dm0,dm0,dm1,r0        // dm0 = (A+B) + B    +  X: ret lr
//   Cycle 12:  nop                              // delay slot 5 (vadd.f-Latenz)
//   Cycle 13:  S: vst.conv cml0[p2,#0]          // delay slot 4
//   Cycle 14:  S: vst.conv cmh0[p2,#64]         // delay slot 3
//   Cycle 15:  nop                              // delay slot 2
//   Cycle 16:  nop                              // delay slot 1
// ⇒ 16 VLIW-Zyklen total.

  // Cycle 1: erste Load + sign-mask in r0 parallel
  vlda.conv.fp32.bf16 cml0, [p0, #0];   movx r0, #60
  // Cycle 2
  vlda.conv.fp32.bf16 cmh0, [p0, #64]
  // Cycle 3
  vlda.conv.fp32.bf16 cml1, [p1, #0]
  // Cycle 4
  vlda.conv.fp32.bf16 cmh1, [p1, #64]
  // Cycles 5..8: warten auf Load-Latenz (5) des letzten Loads
  nop
  nop
  nop
  nop
  // Cycle 9: dm0 = A + B
  vadd.f dm0, dm0, dm1, r0
  // Cycle 10: vadd.f-Latenz
  nop
  // Cycle 11: dm0 = (A+B)+B + ret lr im selben Bundle (V + X)
  vadd.f dm0, dm0, dm1, r0;             ret lr
  // 5 Delay-Slots nach ret:
  nop                                                       // Delay Slot 5 (vadd.f-Latenz)
  vst.conv.bf16.fp32 cml0, [p2, #0]                          // Delay Slot 4
  vst.conv.bf16.fp32 cmh0, [p2, #64]                         // Delay Slot 3
  nop                                                        // Delay Slot 2
  nop                                                        // Delay Slot 1
.Lfunc_end0:
  .size custom_vadd, .Lfunc_end0-custom_vadd
