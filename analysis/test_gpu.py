from runtime.operator import LinearOperator
from runtime.gpu import GPUBackend



op = LinearOperator(
    "test_fc1",
    384,
    1536
)


gpu = GPUBackend()


latency = gpu.execute(op)


print(
    "GPU latency:",
    latency,
    "s"
)