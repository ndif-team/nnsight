### Qwen2.5-7B-Instruct (14.2 GB)

Best wall time per path (s), speedup vs HF mmap in parentheses; GB/s = model size / wall.

| GPU | storage | cache | HF from_pretrained (mmap) | HF from_pretrained (pread) | run:ai stream (CPU clone) | run:ai GPU-direct | run:ai GPU-direct lazy |
|---|---|---|---|---|---|---|---|
| A40 | /work/nvme (Lustre) | cold | 16.8 s, 0.84 GB/s | 17.5 s (0.96x) | 6.9 s (2.44x) | 5.4 s (3.09x) | 5.2 s (3.21x) |
| A40 | /work/nvme (Lustre) | warm | 9.6 s, 1.48 GB/s | 6.3 s (1.51x) | 4.2 s (2.27x) | 2.7 s (3.48x) | 2.6 s (3.68x) |
| A40 | /tmp (local NVMe) | cold | 11.9 s, 1.19 GB/s | 11.4 s (1.04x) | 4.8 s (2.48x) | 3.1 s (3.83x) | 3.2 s (3.76x) |
| A40 | /tmp (local NVMe) | warm | 2.6 s, 5.47 GB/s | 6.0 s (0.43x) | 4.2 s (0.62x) | 2.6 s (1.01x) | 2.5 s (1.04x) |
| A100 | /work/nvme (Lustre) | cold | 16.9 s, 0.84 GB/s | 17.3 s (0.98x) | 7.0 s (2.41x) | 5.4 s (3.12x) | 5.4 s (3.12x) |
| A100 | /work/nvme (Lustre) | warm | 9.5 s, 1.49 GB/s | 6.2 s (1.54x) | 4.4 s (2.17x) | 2.8 s (3.39x) | 2.9 s (3.33x) |
| A100 | /tmp (local NVMe) | cold | 9.3 s, 1.52 GB/s | 10.3 s (0.90x) | 4.9 s (1.90x) | 3.3 s (2.82x) | 3.3 s (2.80x) |
| A100 | /tmp (local NVMe) | warm | 2.6 s, 5.45 GB/s | 5.9 s (0.44x) | 4.3 s (0.61x) | 2.7 s (0.96x) | 2.7 s (0.95x) |
| H200 | /work/nvme (Lustre) | cold | 8.4 s, 1.69 GB/s | 11.8 s (0.71x) | 3.9 s (2.17x) | 2.8 s (2.98x) | 3.0 s (2.80x) |
| H200 | /work/nvme (Lustre) | warm | 5.5 s, 2.58 GB/s | 7.9 s (0.70x) | 2.9 s (1.89x) | 2.0 s (2.81x) | 2.0 s (2.81x) |
| H200 | /tmp (local NVMe) | cold | 12.8 s, 1.11 GB/s | 10.2 s (1.26x) | 3.9 s (3.25x) | 2.8 s (4.60x) | 2.8 s (4.61x) |
| H200 | /tmp (local NVMe) | warm | 2.3 s, 6.29 GB/s | 7.8 s (0.29x) | 2.8 s (0.79x) | 1.9 s (1.17x) | 2.0 s (1.15x) |

### Qwen3-30B-A3B (56.9 GB)

Best wall time per path (s), speedup vs HF mmap in parentheses; GB/s = model size / wall.

| GPU | storage | cache | HF from_pretrained (mmap) | HF from_pretrained (pread) | run:ai stream (CPU clone) | run:ai GPU-direct | run:ai GPU-direct lazy |
|---|---|---|---|---|---|---|---|
| A40 | /work/nvme (Lustre) | cold | 105.1 s, 0.54 GB/s | 62.8 s (1.67x) | 36.7 s (2.86x) | 40.8 s (2.58x) | 28.5 s (3.68x) |
| A40 | /work/nvme (Lustre) | warm | 96.4 s, 0.59 GB/s | 14.5 s (6.65x) | 22.0 s (4.37x) | 14.0 s (6.88x) | 42.1 s (2.29x) |
| A40 | /tmp (local NVMe) | cold | 42.9 s, 1.33 GB/s | 36.6 s (1.17x) | 23.4 s (1.83x) | 15.5 s (2.77x) | 15.7 s (2.73x) |
| A40 | /tmp (local NVMe) | warm | 12.5 s, 4.54 GB/s | 15.1 s (0.83x) | 16.9 s (0.74x) | 12.1 s (1.04x) | 12.5 s (1.00x) |
| A100 | /work/nvme (Lustre) | cold | 103.6 s, 0.55 GB/s | 60.3 s (1.72x) | 39.0 s (2.66x) | 42.4 s (2.44x) | 30.1 s (3.44x) |
| A100 | /work/nvme (Lustre) | warm | 43.3 s, 1.31 GB/s | 14.4 s (3.01x) | 18.3 s (2.37x) | 13.7 s (3.16x) | 13.7 s (3.17x) |
| A100 | /tmp (local NVMe) | cold | 42.6 s, 1.33 GB/s | 36.2 s (1.18x) | 20.9 s (2.04x) | 15.5 s (2.74x) | 15.6 s (2.73x) |
| A100 | /tmp (local NVMe) | warm | 12.1 s, 4.69 GB/s | 15.0 s (0.81x) | 17.4 s (0.70x) | 13.0 s (0.93x) | 13.1 s (0.93x) |
| H200 | /work/nvme (Lustre) | cold | 77.7 s, 0.73 GB/s | 37.3 s (2.09x) | 22.7 s (3.43x) | 17.7 s (4.39x) | 17.3 s (4.49x) |
| H200 | /work/nvme (Lustre) | warm | 24.3 s, 2.34 GB/s | 19.7 s (1.24x) | 16.8 s (1.45x) | 10.9 s (2.23x) | 10.9 s (2.23x) |
| H200 | /tmp (local NVMe) | cold | 61.8 s, 0.92 GB/s | 31.6 s (1.95x) | 17.4 s (3.55x) | 11.5 s (5.37x) | 11.4 s (5.42x) |
| H200 | /tmp (local NVMe) | warm | 11.2 s, 5.07 GB/s | 20.4 s (0.55x) | 17.1 s (0.66x) | 10.5 s (1.07x) | 10.4 s (1.07x) |

### Qwen3-32B (61.0 GB)

Best wall time per path (s), speedup vs HF mmap in parentheses; GB/s = model size / wall.

| GPU | storage | cache | HF from_pretrained (mmap) | HF from_pretrained (pread) | run:ai stream (CPU clone) | run:ai GPU-direct | run:ai GPU-direct lazy |
|---|---|---|---|---|---|---|---|
| A40 | /work/nvme (Lustre) | cold | 63.3 s, 0.96 GB/s | 87.7 s (0.72x) | 31.9 s (1.98x) | 25.2 s (2.51x) | 25.9 s (2.44x) |
| A40 | /work/nvme (Lustre) | warm | 41.9 s, 1.46 GB/s | 27.0 s (1.56x) | 31.4 s (1.33x) | 33.4 s (1.26x) | 33.4 s (1.25x) |
| A40 | /tmp (local NVMe) | cold | 40.7 s, 1.50 GB/s | 53.3 s (0.76x) | 19.8 s (2.06x) | 13.6 s (2.99x) | 13.7 s (2.98x) |
| A40 | /tmp (local NVMe) | warm | 10.0 s, 6.10 GB/s | 86.1 s (0.12x) | 17.4 s (0.58x) | 9.8 s (1.02x) | 9.8 s (1.02x) |
| A100 | /work/nvme (Lustre) | cold | 61.9 s, 0.99 GB/s | 101.0 s (0.61x) | 34.2 s (1.81x) | 27.1 s (2.29x) | 31.3 s (1.98x) |
| A100 | /work/nvme (Lustre) | warm | 45.7 s, 1.34 GB/s | 43.6 s (1.05x) | 41.5 s (1.10x) | 32.5 s (1.40x) | 43.9 s (1.04x) |
| A100 | /tmp (local NVMe) | cold | 40.9 s, 1.49 GB/s | 56.2 s (0.73x) | 19.9 s (2.06x) | 13.8 s (2.97x) | 14.7 s (2.79x) |
| A100 | /tmp (local NVMe) | warm | 10.6 s, 5.77 GB/s | 87.2 s (0.12x) | 21.3 s (0.50x) | 11.6 s (0.91x) | 9.9 s (1.07x) |
| H200 | /work/nvme (Lustre) | cold | 36.6 s, 1.67 GB/s | 51.0 s (0.72x) | 16.7 s (2.20x) | 14.5 s (2.53x) | 22.0 s (1.66x) |
| H200 | /work/nvme (Lustre) | warm | 26.4 s, 2.31 GB/s | 31.0 s (0.85x) | 10.7 s (2.46x) | 9.0 s (2.94x) | 9.8 s (2.70x) |
| H200 | /tmp (local NVMe) | cold | 61.5 s, 0.99 GB/s | 41.4 s (1.48x) | 15.9 s (3.87x) | 13.3 s (4.63x) | 11.7 s (5.26x) |
| H200 | /tmp (local NVMe) | warm | 10.0 s, 6.10 GB/s | 31.4 s (0.32x) | 10.7 s (0.94x) | 9.3 s (1.07x) | 9.3 s (1.08x) |
