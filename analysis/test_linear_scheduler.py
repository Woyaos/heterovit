from analysis.build_linear import linear_ops

from runtime.scheduler.linear_scheduler import LinearScheduler

from runtime.cost_model.gpu import GPUModel

from runtime.cost_model.fpga import FPGAModel

from runtime.cost_model.communication import CommunicationModel


gpu=GPUModel()

fpga=FPGAModel()

comm=CommunicationModel()


scheduler=LinearScheduler(
    gpu,
    fpga,
    comm
)



gpu_total=0

fpga_total=0



for op in linear_ops:


    gpu_time=scheduler.gpu_only(
        {
            "MAC":op.mac
        }
    )


    fpga_time=scheduler.fpga_offload(
        {
            "MAC":op.mac,
            "activation_size":op.activation_size
        }
    )


    gpu_total+=gpu_time

    fpga_total+=fpga_time



print("================")

print(
    "GPU only:",
    gpu_total
)


print(
    "Linear FPGA:",
    fpga_total
)