class FPGAFrontend:


    def __init__(
            self,
            throughput=100e12,
            bandwidth=16e9
    ):

        # FPGA bit计算吞吐
        self.throughput = throughput


        # PCIe带宽
        self.bandwidth = bandwidth



    def compute_time(self, operator):

        return (
            operator.mac /
            self.throughput
        )


    def transfer_time(
            self,
            operator
    ):

        # 输入输出数据量

        input_bytes = (
            operator.in_features
            *4
        )


        output_bytes = (
            operator.out_features
            *4
        )


        total_bytes = (
            input_bytes+
            output_bytes
        )


        return (
            total_bytes /
            self.bandwidth
        )



    def execute(self,operator):

        compute = self.compute_time(operator)


        transfer = self.transfer_time(operator)


        return compute+transfer