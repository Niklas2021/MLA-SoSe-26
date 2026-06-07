module {
  aie.device(npu2) {
    func.func private @matmul(memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) attributes {link_with = "matmul.o"}
    func.func private @zero(memref<2x2x8x8xbf16>) attributes {link_with = "zero.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
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
          %buffer_out = aie.objectfifo.acquire @out_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()

          scf.for %arg1 = %c0 to %c16 step %c1 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_0(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
           }
          aie.objectfifo.release @out_L1L2_0_0(Produce, 1)
          }

      aie.end
    } {stack_size = 1024 : i32}
    aie.runtime_sequence(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x128xbf16>, %arg2: memref<256x128xbf16>) {
      // Non-blocking data movement (Task 4): 8 groups of 2 M-tile rows (a), ping-pong
      // over TWO disjoint BD sets, so while waiting on one group the next is already
      // in flight (no blocking stall).
      //   Set A = bd {0=out, 1,3=in0, 2,4=in1}   (even groups)
      //   Set B = bd {8=out, 9,11=in0, 10,12=in1} (odd groups)  -> max 10 of 16 BDs
      // Per group: out (2 a-rows x 8 b), and per a: in0 (b-repeat) + in1 (re-sent).
      // dma_wait(out) is DEFERRED: issue the next group first, then wait on the group
      // from 2 ago (whose BD set is reused) -> the wait overlaps with the already-issued
      // group. Only the last two waits block.

      // --- Group 0 (Set A): a = 0,1 ---
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][2, 8, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][8, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 16384][8, 16, 16, 64][0, 64, 1024, 1]) {id = 3 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>

      // --- Group 1 (Set B): a = 2,3 --- (issued before waiting on group 0)
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 4096][2, 8, 16, 16][2048, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 32768][8, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 10 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 49152][8, 16, 16, 64][0, 64, 1024, 1]) {id = 11 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 12 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>

      // --- Group 2 (Set A): a = 4,5 --- (wait on group 0 first -> Set A free)
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 8192][2, 8, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 65536][8, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 81920][8, 16, 16, 64][0, 64, 1024, 1]) {id = 3 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>

      // --- Group 3 (Set B): a = 6,7 --- (wait on group 1 first -> Set B free)
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 12288][2, 8, 16, 16][2048, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 98304][8, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 10 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 114688][8, 16, 16, 64][0, 64, 1024, 1]) {id = 11 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 12 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>

      // --- Group 4 (Set A): a = 8,9 ---
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 16384][2, 8, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 131072][8, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 147456][8, 16, 16, 64][0, 64, 1024, 1]) {id = 3 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>

      // --- Group 5 (Set B): a = 10,11 ---
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 20480][2, 8, 16, 16][2048, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 163840][8, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 10 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 180224][8, 16, 16, 64][0, 64, 1024, 1]) {id = 11 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 12 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>

      // --- Group 6 (Set A): a = 12,13 ---
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 24576][2, 8, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 196608][8, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 212992][8, 16, 16, 64][0, 64, 1024, 1]) {id = 3 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>

      // --- Group 7 (Set B): a = 14,15 ---
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 28672][2, 8, 16, 16][2048, 16, 128, 1]) {id = 8 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 229376][8, 16, 16, 64][0, 64, 1024, 1]) {id = 9 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 10 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 245760][8, 16, 16, 64][0, 64, 1024, 1]) {id = 11 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 16, 64, 16][16, 8192, 128, 1]) {id = 12 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>

      // last two waits (blocking; nothing left to overlap)
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
    }
  }
}
