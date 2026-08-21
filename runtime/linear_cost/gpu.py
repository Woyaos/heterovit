class GPUBackend:


    def __init__(self, throughput=10e12):

        # 假设GPU计算能力
        # 10 TOPS
        self.throughput = throughput



    def compute_time(self, operator):

        """
        根据MAC估算GPU计算时间
        """

        time = (
            operator.mac /
            self.throughput
        )


        return time



    def execute(self, operator):

        latency = self.compute_time(operator)


        return latency