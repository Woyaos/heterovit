class ViTExecutor:


    def __init__(
        self,
        blocks,
        gpu_model,
        fpga_model,
        comm_model
    ):

        self.blocks = blocks

        self.gpu = gpu_model
        self.fpga = fpga_model
        self.comm = comm_model



    def run_gpu_only(self):

        total_latency = 0


        for block in self.blocks:


            # attention GPU
            attention_latency = self.gpu.latency(
                block.attention_compute()
            )


            # FFN GPU
            ffn_latency = self.gpu.latency(
                block.ffn_compute()
            )


            total_latency += (
                attention_latency
                +
                ffn_latency
            )


        return total_latency

    def run_block_offload(self):


        total_latency = 0


        for block in self.blocks:


            # Attention GPU

            attention_time = self.gpu.latency(
                block.attention_compute()
            )


            # FFN FPGA
            # 包含:
            # GPU->FPGA
            # FPGA计算
            # FPGA->GPU

            ffn_time = self.fpga.total_latency(
                block.ffn_compute(),
                block.activation_size
            )


            block_latency = (
                attention_time
                +
                ffn_time
            )


            total_latency += block_latency



        return total_latency