import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one thread per process: runs are parallel processes
  os.environ.setdefault(_v, "1")
