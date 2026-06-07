module {
  aie.device(npu2) {
    func.func private @matmul(memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xf32>) attributes {link_with = "matmul.o"}
    func.func private @zero(memref<2x2x8x8xf32>) attributes {link_with = "zero.o"}
    func.func private @convert(memref<2x2x8x8xf32>, memref<2x2x8x8xbf16>) attributes {link_with = "convert.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %acc = aie.buffer(%tile_0_2) {sym_name = "acc"} : memref<2x2x8x8xf32>
    aie.objectfifo @in0_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo.link [@in0_L3L2_0] -> [@in0_L2L1_0]([] [])
    aie.objectfifo @in1_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo @in1_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 2, stride = 8>, <size = 8, stride = 16>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8x2x8x8xbf16>>
    aie.objectfifo.link [@in1_L3L2_0] -> [@in1_L2L1_0]([] [])
    aie.objectfifo @out_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L2L3_0(%mem_tile_0_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16x16xbf16>>
    aie.objectfifo.link [@out_L1L2_0_0] -> [@out_L2L3_0]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      scf.for %arg0 = %c0 to %c128 step %c1 {
          // Zero the FP32 accumulator, accumulate all c in full precision,
          // then convert once to bf16 into the output FIFO.
          func.call @zero(%acc) : (memref<2x2x8x8xf32>) -> ()

          scf.for %arg1 = %c0 to %c16 step %c1 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %acc) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xf32>) -> ()
            aie.objectfifo.release @in0_L2L1_0(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
           }

          %buffer_out = aie.objectfifo.acquire @out_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @convert(%acc, %out) : (memref<2x2x8x8xf32>, memref<2x2x8x8xbf16>) -> ()
          aie.objectfifo.release @out_L1L2_0_0(Produce, 1)
          }
      
      aie.end
    } {stack_size = 1024 : i32}
    aie.runtime_sequence(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x128xbf16>, %arg2: memref<256x128xbf16>) {
      // Datenbewegung in Bloecken von 4 M-Tile-Zeilen (a). Pro Block:
      //   out (C): 1 Deskriptor (bd 0) ueber 4 a-Zeilen x 8 b ; Layout pqmn -> MN
      //     sizes [a=4, b=8, pm=16, qn=16]  strides [M-row*16=2048, qn-blk=16, M-row=128, 1]
      //   in0 (A): pro a ein Deskriptor (bd 1,3,5,7). b-Repeat = stride 0 (aeusserste Dim!)
      //     sizes [b=8(rep), c=16, pm=16, rk=64]  strides [0, c*64=64, M-row=1024, 1] ; off a*16384
      //   in1 (B): pro a erneut gesendet (= Repeat ueber a) (bd 2,4,6,8)
      //     sizes [b=8, c=16, kr=64, nc=16]  strides [b-blk=16, c*64*N=8192, K-row=128, 1]
      // 16 BDs total: hier nur bd 0..8. dma_wait(out) am Blockende ist die Barriere,
      // die garantiert, dass die Inputs des Blocks konsumiert sind, bevor bd 0..8 neu belegt werden.

      // --- Block 0 : a = 0..3 ---
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][4, 8, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][8, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 16384][8, 16, 16, 64][0, 64, 1024, 1]) {id = 3 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 32768][8, 16, 16, 64][0, 64, 1024, 1]) {id = 5 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 6 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 49152][8, 16, 16, 64][0, 64, 1024, 1]) {id = 7 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 8 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_wait {symbol = @out_L2L3_0}

      // --- Block 1 : a = 4..7 ---
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 8192][4, 8, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 65536][8, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 81920][8, 16, 16, 64][0, 64, 1024, 1]) {id = 3 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 98304][8, 16, 16, 64][0, 64, 1024, 1]) {id = 5 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 6 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 114688][8, 16, 16, 64][0, 64, 1024, 1]) {id = 7 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 8 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_wait {symbol = @out_L2L3_0}

      // --- Block 2 : a = 8..11 ---
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 16384][4, 8, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 131072][8, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 147456][8, 16, 16, 64][0, 64, 1024, 1]) {id = 3 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 163840][8, 16, 16, 64][0, 64, 1024, 1]) {id = 5 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 6 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 180224][8, 16, 16, 64][0, 64, 1024, 1]) {id = 7 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 8 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_wait {symbol = @out_L2L3_0}

      // --- Block 3 : a = 12..15 ---
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 24576][4, 8, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 196608][8, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 212992][8, 16, 16, 64][0, 64, 1024, 1]) {id = 3 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 229376][8, 16, 16, 64][0, 64, 1024, 1]) {id = 5 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 6 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 245760][8, 16, 16, 64][0, 64, 1024, 1]) {id = 7 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 8 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
    }
  }
}