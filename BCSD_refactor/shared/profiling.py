import torch
import time
from typing import Dict


class InferenceProfiler:
    """Tracks asynchronous CUDA time and peak memory."""
    def __init__(self, device: str):
        self.device = device
        self.is_cuda = 'cuda' in device
        if self.is_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_time_ms = 0.0
        self.peak_memory_mb = 0.0
        self.total_samples = 0

    def __enter__(self):
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_event.record()
        else:
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            self.total_time_ms += self.start_event.elapsed_time(self.end_event)
            peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            self.peak_memory_mb = max(self.peak_memory_mb, peak_mem)
        else:
            self.total_time_ms += (time.time() - self.start_time) * 1000

    def get_stats(self) -> Dict[str, float]:
        return {
            "avg_ms_per_sample": self.total_time_ms / max(1, self.total_samples),
            "peak_memory_mb": self.peak_memory_mb
        }
