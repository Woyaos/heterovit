from .communication import CommunicationModel



class FPGAModel:


    def __init__(
        self,
        throughput=100e12,
        bandwidth=16e9
    ):

        self.throughput = throughput

        self.communication = CommunicationModel(
            bandwidth
        )



    def compute_latency(
        self,
        mac
    ):

        return (
            mac /
            self.throughput
        )



    def total_latency(
        self,
        mac,
        activation_size
    ):

        compute = self.compute_latency(
            mac
        )


        communication = (
            2 *
            self.communication.latency(
                activation_size
            )
        )


        return (
            compute +
            communication
        )