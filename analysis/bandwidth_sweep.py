from runtime.operator import LinearOperator
from runtime.cost_model.fpga import FPGAModel


op = LinearOperator(
    "fc1",
    384,
    1536
)


bandwidths=[
    4e9,
    8e9,
    16e9,
    32e9,
    64e9,
    128e9
]


print("MAC:",op.mac)

print(
    "Activation:",
    op.activation_size
)



for bw in bandwidths:


    fpga = FPGAModel(
        bandwidth=bw
    )


    t = fpga.total_latency(
        op.mac,
        op.activation_size
    )


    print(
        bw/1e9,
        "GB/s:",
        t
    )