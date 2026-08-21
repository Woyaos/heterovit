from runtime.operator import LinearOperator
from runtime.fpga import FPGAFrontend



op = LinearOperator(
    "fc1",
    384,
    1536
)


fpga = FPGAFrontend()


print(
    "FPGA latency:",
    fpga.execute(op)
)