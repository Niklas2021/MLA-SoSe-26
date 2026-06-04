  .file "matmul.s"
  .section .text.matmul,"ax",@progbits
  .globl matmul
  .p2align 4
  .type matmul,@function
matmul:
// Computes out += in0 * in1   (out is zero-initialised)  -- PIPELINED version.
// L1 views (p=2,q=2,r=8,m=8,n=8,k=8): in0 prmk (mk), in1 rqkn (kn->nk), out pqmn (mn).
//   out[p][q] = sum_r in0[p][r] @ in1[r][q]
//
// Software-pipelined over r (II = 16): loads of iteration r+1 are prefetched into
// the latency shadows of iteration r. vconv ex0 (in0) is scheduled AFTER the four
// shuffles so the q0/q1 conversion chains lag by only 2 (not 3) -> the two vmacs
// are 2 apart. Each pass keeps out[p][0]=dm0, out[p][1]=dm1.
//
// Register map:
//   ones (bf16 1.0)         y5 = (x10,x11)
//   in0 fp32 -> dm2 -> ex0(=x0)
//   q0 in1: raw x2,x4 -> shuffle x6,x7(=y3) -> fp32 dm3 -> ex6(=x6, reuses shuffle-lo)
//   q1 in1: raw x3,x5 -> shuffle x8,x9(=y4) -> fp32 dm4 -> ex8(=x8, reuses shuffle-lo)
//   outputs dm0 (q=0), dm1 (q=1)
//   modes r4=#52,r5=#53 (shuffle), r6=#60 (vmul), r7=#780 (vmac)
//
// Per-r steady body (16 cycles); prefetch = loads for r+1:
//   c0  M vshuffle x6,x2,x4
//   c1  M vshuffle x7,x2,x4
//   c2  M vshuffle x8,x3,x5 | B vldb x2            (prefetch, after c0/c1 read x2)
//   c3  M vshuffle x9,x3,x5 | V vmul dm3 | B vldb x4   (prefetch x4)
//   c4  M vconv ex0,dm2                             (in0; reads dm2 before prefetch overwrites)
//   c5  V vmul dm4          | A vlda cml2           (prefetch dm2-lo, after c4 read dm2)
//   c6  A vlda cmh2         | B vldb x3             (prefetch dm2-hi, x3)
//   c7  B vldb x5
//   c8  nop
//   c9  M vconv ex6,dm3      (dm3 ready c3+6=9)
//   c10 nop
//   c11 M vconv ex8,dm4      (dm4 ready c5+6=11)
//   c12 nop
//   c13 V vmac dm0,dm0,ex0,ex6   (ex0 ready c4+4=8 ; ex6 ready c9+4=13)
//   c14 nop
//   c15 V vmac dm1,dm1,ex0,ex8   (ex8 ready c11+4=15)

  movxm r3, #16256                 // 0x3F80 = bf16(1.0)
  vbcst.16 x10, r3
  vmov x11, x10
  mova  r4, #52
  mova  r5, #53
  mova  r6, #60
  movxm r7, #780
  mov   p3, p1                     // save in1 base

  //===================================================================
  // Pass p=0 : out[0][0]=dm0, out[0][1]=dm1
  //===================================================================
  mov p1, p3
  // prologue: load r=0 inputs (prefetch) + the two out tiles
  vlda.conv.fp32.bf16 cml2, [p0], #64;  vldb x2, [p1], #64
  vlda.conv.fp32.bf16 cmh2, [p0], #64;  vldb x4, [p1], #64
  vldb x3, [p1], #64
  vldb x5, [p1], #64
  vlda.conv.fp32.bf16 cml0, [p2, #0]
  vlda.conv.fp32.bf16 cmh0, [p2, #64]
  vlda.conv.fp32.bf16 cml1, [p2, #128]
  vlda.conv.fp32.bf16 cmh1, [p2, #192]
  // steady state r=0..6 (compute r, prefetch r+1)
  .rept 7
    vshuffle x6, x2, x4, r4
    vshuffle x7, x2, x4, r5
    vshuffle x8, x3, x5, r4;                             vldb x2, [p1], #64
    vshuffle x9, x3, x5, r5;  vmul.f dm3, y3, y5, r6;    vldb x4, [p1], #64
    vconv.bfp16ebs8.fp32 ex0, dm2
    vmul.f dm4, y4, y5, r6;    vlda.conv.fp32.bf16 cml2, [p0], #64
    vlda.conv.fp32.bf16 cmh2, [p0], #64;                vldb x3, [p1], #64
    vldb x5, [p1], #64
    nop
    vconv.bfp16ebs8.fp32 ex6, dm3
    nop
    vconv.bfp16ebs8.fp32 ex8, dm4
    nop
    vmac.f dm0, dm0, ex0, ex6, r7
    nop
    vmac.f dm1, dm1, ex0, ex8, r7
  .endr
  // peeled last iteration r=7 (compute only, no prefetch -> pointers stay aligned)
  vshuffle x6, x2, x4, r4
  vshuffle x7, x2, x4, r5
  vshuffle x8, x3, x5, r4
  vshuffle x9, x3, x5, r5;  vmul.f dm3, y3, y5, r6
  vconv.bfp16ebs8.fp32 ex0, dm2
  vmul.f dm4, y4, y5, r6
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex6, dm3
  nop
  vconv.bfp16ebs8.fp32 ex8, dm4
  nop
  vmac.f dm0, dm0, ex0, ex6, r7
  nop
  vmac.f dm1, dm1, ex0, ex8, r7
  // epilogue: store out[0][0], out[0][1]
  nop
  nop
  nop
  vst.conv.bf16.fp32 cml0, [p2], #64
  vst.conv.bf16.fp32 cmh0, [p2], #64
  vst.conv.bf16.fp32 cml1, [p2], #64
  vst.conv.bf16.fp32 cmh1, [p2], #64

  //===================================================================
  // Pass p=1 : out[1][0]=dm0, out[1][1]=dm1
  //  (p0 already at in0[1][0]; p2 already at out[1][0]; in1 reset below)
  //===================================================================
  mov p1, p3
  vlda.conv.fp32.bf16 cml2, [p0], #64;  vldb x2, [p1], #64
  vlda.conv.fp32.bf16 cmh2, [p0], #64;  vldb x4, [p1], #64
  vldb x3, [p1], #64
  vldb x5, [p1], #64
  vlda.conv.fp32.bf16 cml0, [p2, #0]
  vlda.conv.fp32.bf16 cmh0, [p2, #64]
  vlda.conv.fp32.bf16 cml1, [p2, #128]
  vlda.conv.fp32.bf16 cmh1, [p2, #192]
  .rept 7
    vshuffle x6, x2, x4, r4
    vshuffle x7, x2, x4, r5
    vshuffle x8, x3, x5, r4;                             vldb x2, [p1], #64
    vshuffle x9, x3, x5, r5;  vmul.f dm3, y3, y5, r6;    vldb x4, [p1], #64
    vconv.bfp16ebs8.fp32 ex0, dm2
    vmul.f dm4, y4, y5, r6;    vlda.conv.fp32.bf16 cml2, [p0], #64
    vlda.conv.fp32.bf16 cmh2, [p0], #64;                vldb x3, [p1], #64
    vldb x5, [p1], #64
    nop
    vconv.bfp16ebs8.fp32 ex6, dm3
    nop
    vconv.bfp16ebs8.fp32 ex8, dm4
    nop
    vmac.f dm0, dm0, ex0, ex6, r7
    nop
    vmac.f dm1, dm1, ex0, ex8, r7
  .endr
  vshuffle x6, x2, x4, r4
  vshuffle x7, x2, x4, r5
  vshuffle x8, x3, x5, r4
  vshuffle x9, x3, x5, r5;  vmul.f dm3, y3, y5, r6
  vconv.bfp16ebs8.fp32 ex0, dm2
  vmul.f dm4, y4, y5, r6
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex6, dm3
  nop
  vconv.bfp16ebs8.fp32 ex8, dm4
  nop
  vmac.f dm0, dm0, ex0, ex6, r7
  nop
  vmac.f dm1, dm1, ex0, ex8, r7
  nop
  nop
  nop
  vst.conv.bf16.fp32 cml0, [p2], #64
  vst.conv.bf16.fp32 cmh0, [p2], #64
  vst.conv.bf16.fp32 cml1, [p2], #64
  vst.conv.bf16.fp32 cmh1, [p2], #64

  ret lr
  nop                                 // Delay Slot 5
  nop                                 // Delay Slot 4
  nop                                 // Delay Slot 3
  nop                                 // Delay Slot 2
  nop                                 // Delay Slot 1
.Lfunc_end0:
  .size matmul, .Lfunc_end0-matmul
