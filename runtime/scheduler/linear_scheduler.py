class LinearScheduler:


    def __init__(
        self,
        gpu_model,
        fpga_model,
        comm_model
    ):

        self.gpu = gpu_model
        self.fpga = fpga_model
        self.comm = comm_model



    def gpu_only(
        self,
        linear
    ):

        return self.gpu.latency(
            linear["MAC"]
        )



    def fpga_offload(
        self,
        linear
    ):


        compute = self.fpga.latency(
            linear["MAC"]
        )


        communication = self.comm.latency(
            linear["activation_size"]
        )


        return (
            2*communication
            +
            compute
        )