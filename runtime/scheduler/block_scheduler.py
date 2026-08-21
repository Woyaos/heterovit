class BlockScheduler:


    def __init__(
        self,
        cost_model
    ):

        self.cost_model = cost_model



    def gpu_only(
        self,
        block
    ):


        cost = self.cost_model.estimate(block)


        return (
            cost["attention_gpu"]
            +
            cost["ffn_gpu"]
        )



    def ffn_fpga(
        self,
        block
    ):


        cost = self.cost_model.estimate(block)


        return (
            cost["attention_gpu"]
            +
            2*cost["communication"]
            +
            cost["ffn_fpga"]
        )