class CommunicationModel:


    def __init__(
        self,
        bandwidth=16e9
    ):

        self.bandwidth=bandwidth



    def latency(
        self,
        bytes
    ):

        return bytes/self.bandwidth