#!/usr/bin/env python
"""Pre-build all Numba JIT caches.

Run once after installation to eliminate JIT latency:
    python prebuild_cache.py
"""

from simulation.warmup import warmup

if __name__ == "__main__":
    warmup()
    print("All JIT caches built successfully.")
