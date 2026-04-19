# Task 1: GPU Device Properties
# Report L2CacheSize, MaxSharedMemoryPerMultiprocessor, ClockRate via cp.cuda.Device().attributes

import cupy as cp

# return dictionary of attribute values with the names as keys
attr = dict(cp.cuda.Device().attributes.items())

print("L2CacheSize:", attr["L2CacheSize"])
print("MaxSharedMemoryPerMultiprocessor:", attr["MaxSharedMemoryPerMultiprocessor"])
print("ClockRate:", attr["ClockRate"])
