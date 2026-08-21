from runtime.gpu import GPUBackend
from runtime.fpga import FPGAFrontend



class Scheduler:


    def __init__(self):

        self.gpu = GPUBackend()

        self.fpga = FPGAFrontend()



    def decide(self, operator):


        gpu_time = (
            self.gpu.execute(operator)
        )


        fpga_time = (
            self.fpga.execute(operator)
        )


        if fpga_time < gpu_time:

            operator.device="FPGA"

        else:

            operator.device="GPU"



        return operator