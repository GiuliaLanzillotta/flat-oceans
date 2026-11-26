# Hardware and resource requirements

## Reference environment

The experiments used the shared Linux server documented in the paper's **Computing infrastructure** paragraph (Appendix: Experimental methods):

| Component | Recorded setup |
|---|---|
| CPU | 2 × AMD EPYC 9554: 128 physical cores, 256 hardware threads |
| Host memory | Approximately 377 GiB usable RAM |
| GPU | 4 × NVIDIA RTX A6000, 48 GB VRAM per GPU |
| Software | Linux x86-64, Python 3.10.12, PyTorch 2.5.1 with CUDA 12.4 |
| Numerical precision | Float64 for geometry calculations; training precision is recorded per run |

The core counts and four GPUs describe the shared server, not a requirement to allocate the entire machine to one experiment. Dense Jacobian QR/SVD and eigensystem calculations also use CPU linear algebra and substantial host RAM; GPU memory does not replace that RAM.

## What to provision

These are planning budgets, **not benchmarked minimum specifications**. No systematic minimum-memory study or timed, clean end-to-end reproduction has been performed.

| Task | Resources |
|---|---|
| Compile the paper and create the arXiv ZIP | CPU only; a normal laptop/desktop with a TeX installation. No datasets, GPU, or experimental arrays are needed. |
| Run the regression tests | CPU only; tests include small MNIST training/replay runs. The 110-test suite took about 4 minutes on the reference server with two BLAS/OpenMP threads. Download MNIST first. |
| Regenerate figures from saved measurements | CPU plotting. Availability of the required saved measurements is the main prerequisite; a fresh clone does not include the large arrays. Some recipes load sizeable saved outputs, so memory use depends on the figure. |
| Reproduce the complete experiment catalog serially | Provision a Linux research server comparable in RAM to the reference machine (about 384 GiB), at least 16 available CPU threads, and a CUDA-capable NVIDIA GPU with 48 GB VRAM available to the job. Budget at least 2 TB of free fast storage for results, datasets, and working files. |

One GPU can execute the serial catalog: the retained programs do not require distributed or multi-GPU execution. Four GPUs were available to run independent work concurrently. A smaller GPU or less host RAM may suffice for selected experiments, but the full suite has not been validated under those constraints. Keep the documented batch sizes, precision, and numerical thresholds when reproducing the reported protocol.

The retained local measurement tree occupies approximately **925 GB (decimal)**, mostly factors and subspace bases. The 2 TB budget leaves space for datasets, intermediate files, and output generation; it is not a measured peak disk bound. The published paper can be built without any of these measurements.

## Execution limits

The reproduction runner executes jobs sequentially and defaults to eight CPU threads. On a suitable dedicated machine, use:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce.py all --execute --threads 16
```

Choose an otherwise available GPU with `CUDA_VISIBLE_DEVICES`; selecting one physical GPU exposes it as `cuda:0` to PyTorch. The code does not pool VRAM across cards. Use a driver compatible with the installed CUDA-enabled PyTorch build; `nvcc` is not needed for the supplied Python programs.

For a shared server, `scripts/jobs.sh` provides admission checks. The original operating policy allowed at most two heavy jobs, used 16 CPU threads per heavy job, and reserved 100 GB of free host-memory headroom beyond declared job estimates. The headroom is a scheduling precaution, not an experimental parameter or a measured per-job requirement. `reproduce.py` itself does not acquire this admission lock; use the wrapper when sharing a machine.

Full reproduction includes 155 training runs and 887 catalog steps, including large matrix factorizations. Plan for sustained server use; the catalog has not been timed end-to-end, so no total runtime guarantee is given. Numerical results can vary across hardware and numerical libraries. Preserve validation failures and follow the recorded gates rather than changing thresholds to force agreement.
