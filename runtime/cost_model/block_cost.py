from .gpu import GPUModel
from .fpga import FPGAModel
from .communication import CommunicationModel



class BlockCostModel:


    def __init__(self):

        self.gpu=GPUModel()

        self.fpga=FPGAModel()

        self.comm=CommunicationModel()



    def estimate(
        self,
        block
    ):


        result={}


        # Attention GPU

        attn_gpu = self.gpu.latency(
            block.attention_compute()
        )


        # FFN GPU

        ffn_gpu=self.gpu.latency(
            block.ffn_compute()
        )


        # FFN FPGA

        ffn_fpga=self.fpga.latency(
            block.ffn_compute()
        )



        # communication

        comm=self.comm.latency(
            block.activation_size
        )



        result["attention_gpu"]=attn_gpu

        result["ffn_gpu"]=ffn_gpu

        result["ffn_fpga"]=ffn_fpga

        result["communication"]=comm


        return result