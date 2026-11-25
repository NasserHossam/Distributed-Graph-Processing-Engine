"""Setup check: every MPI process prints its rank, then they do one collective op."""
from mpi4py import MPI

comm = MPI.COMM_WORLD          # the group of all processes started by mpiexec
rank = comm.Get_rank()         # this process's id: 0 .. size-1
size = comm.Get_size()         # total number of processes

print(f"Hello from rank {rank} of {size} on {MPI.Get_processor_name()}", flush=True)

# Sanity check that communication works: sum all ranks onto every process.
total = comm.allreduce(rank, op=MPI.SUM)
comm.Barrier()                 # wait so rank 0's message prints last
if rank == 0:
    print(f"allreduce check: sum of ranks = {total} (expected {size * (size - 1) // 2})")