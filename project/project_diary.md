
```bash
python3 -m venv venv
source venv/bin/activate
pip install cuda-python
pip install cupy-cuda13x
pip install cuda-tile
```

*Ordner vom lokalen Rechner auf flambe kopieren:*

```bash
scp -r  /Users/dascha/Desktop/MLA-SoSe-26/project mla05@flambe.inf-ra.uni-jena.de:/home/mla05/test```

--------------------------------------
#### Welche GPU Properties sind für uns wichtig?

*SMEM-Budget:*
- **MaxSharedMemoryPerBlockOptin**: 101376 (≈99 KB) — das Budget fürs Pruning, ob Akku-Tile in SMEM passt  
- **MaxSharedMemoryPerBlock**: 49152 (48 KB) — default. nutzt cuTile dynamisch das Opt-in-Limit (99 KB) oder bleibt es bei 48 KB? Das entscheidet, gegen welche Schranke prunen -> in M2 verifizieren
- **ReservedSharedMemoryPerBlock**: 1024 — 1 KB SMEM ist reserviert, nutzbar ist also Optin − 1024. Für ehrliches Pruning abziehen!!


*Grid-Sizing & Tile-Mapping:*
- **MultiProcessorCount**: 48 — um zu prüfen, ob das Grid die SMs überhaupt füllt

- L2 Cache
__________________________________
#### Vefügbare Properties:
```bash
Output:
{'AsyncEngineCount': 1, 
'CanFlushRemoteWrites': 0, 
'CanMapHostMemory': 1, 
'CanUseHostPointerForRegisteredMem': 1, 
'ClockRate': 2418000, 
'ComputeMode': 0, 
'ComputePreemptionSupported': 1,
'ConcurrentKernels': 1, 
'ConcurrentManagedAccess': 1,
'CooperativeLaunch': 1, 
'CooperativeMultiDeviceLaunch': 1, 
'DirectManagedMemAccessFromHost': 0,
'EccEnabled': 0, 
'GPUDirectRDMAFlushWritesOptions': 1, 
'GPUDirectRDMASupported': 0,
'GPUDirectRDMAWritesOrdering': 100, 
'GlobalL1CacheSupported': 1, 
'GlobalMemoryBusWidth': 256, 
'GpuOverlap': 1, 
'HostNativeAtomicSupported': 1, 
'HostRegisterReadOnlySupported': 0, 
'HostRegisterSupported': 1, 
'Integrated': 1,
'IsMultiGpuBoard': 0, 
'KernelExecTimeout': 0, 
'L2CacheSize': 25165824,
'LocalL1CacheSupported': 1, 
'ManagedMemory': 1, 
'MaxBlockDimX': 1024, 
'MaxBlockDimY': 1024, 
'MaxBlockDimZ': 64, 
'MaxBlocksPerMultiprocessor': 24, 
'MaxGridDimX': 2147483647, 
'MaxGridDimY': 65535, 
'MaxGridDimZ': 65535, 
'MaxPitch': 2147483647, 
'MaxRegistersPerBlock': 65536,
'MaxRegistersPerMultiprocessor': 65536, 
'MaxSharedMemoryPerBlock': 49152, 
'MaxSharedMemoryPerBlockOptin': 101376, 
'MaxSharedMemoryPerMultiprocessor': 102400,
'MaxSurface1DLayeredLayers': 2048,'MaxSurface1DLayeredWidth': 32768,
'MaxSurface1DWidth': 32768, 
'MaxSurface2DHeight': 65536, 
'MaxSurface2DLayeredHeight': 32768,
'MaxSurface2DLayeredLayers': 2048,
'MaxSurface2DLayeredWidth': 32768,
'MaxSurface2DWidth': 131072, 
'MaxSurface3DDepth': 16384, 
'MaxSurface3DHeight': 16384, 
'MaxSurface3DWidth': 16384, 
'MaxSurfaceCubemapLayeredLayers': 2046, 
'MaxSurfaceCubemapLayeredWidth': 32768,
'MaxSurfaceCubemapWidth': 32768, 
'MaxTexture1DLayeredLayers': 2048,
'MaxTexture1DLayeredWidth': 32768, 
'MaxTexture1DLinearWidth': 268435456, 
'MaxTexture1DMipmappedWidth': 32768,
'MaxTexture1DWidth': 131072, 
'MaxTexture2DGatherHeight': 32768, 
'MaxTexture2DGatherWidth': 32768, 
'MaxTexture2DHeight': 65536, 
'MaxTexture2DLayeredHeight': 32768, 
'MaxTexture2DLayeredLayers': 2048, 
'MaxTexture2DLayeredWidth': 32768, 
'MaxTexture2DLinearHeight': 65000, 
'MaxTexture2DLinearPitch': 2097120, 
'MaxTexture2DLinearWidth': 131072, 
'MaxTexture2DMipmappedHeight': 32768, 
'MaxTexture2DMipmappedWidth': 32768,
'MaxTexture2DWidth': 131072, 
'MaxTexture3DDepth': 16384, 
'MaxTexture3DDepthAlt': 32768,
'MaxTexture3DHeight': 16384, 
'MaxTexture3DHeightAlt': 8192,
'MaxTexture3DWidth': 16384, 
'MaxTexture3DWidthAlt': 8192, 'MaxTextureCubemapLayeredLayers': 2046, 'MaxTextureCubemapLayeredWidth': 32768, 'MaxTextureCubemapWidth': 32768, 'MaxThreadsPerBlock': 1024, 'MaxThreadsPerMultiProcessor': 1536, 'MaxTimelineSemaphoreInteropSupported': 1, 'MemoryClockRate': 8533000, 'MemoryPoolSupportedHandleTypes': 1, 'MemoryPoolsSupported': 1, 'MultiGpuBoardGroupID': 0, 'MultiProcessorCount': 48, 'PageableMemoryAccess': 1, 'PageableMemoryAccessUsesHostPageTables': 1, 'PciBusId': 1, 'PciDeviceId': 0, 'PciDomainId': 15, 'Reserved92': 0, 'Reserved93': 0, 'Reserved94': 0, 'ReservedSharedMemoryPerBlock': 1024, 'SingleToDoublePrecisionPerfRatio': 64, 'SparseCudaArraySupported': 1, 'StreamPrioritiesSupported': 1, 'SurfaceAlignment': 512, 'TccDriver': 0, 'TextureAlignment': 512, 'TexturePitchAlignment': 32, 'TotalConstantMemory': 65536, 'UnifiedAddressing': 1, 'WarpSize': 32}

```