class GPUModel:


    def __init__(
        self,
        throughput=10e12
    ):
        # MAC/s
        self.throughput = throughput



    def latency(
        self,
        mac
    ):

        return mac / self.throughput