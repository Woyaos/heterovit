class BlockOperator:


    def __init__(self,name):

        self.name=name


        # Attention

        self.qkv_mac=0

        self.proj_mac=0

        self.attention_mac=0



        # FFN

        self.fc1_mac=0

        self.fc2_mac=0



        # activation大小
        # bytes

        self.activation_size = (
            196 * 384 * 2
        )



        # scheduler结果

        self.attention_device="GPU"

        self.ffn_device="GPU"



    def attention_compute(self):

        return (
            self.qkv_mac
            +
            self.proj_mac
            +
            self.attention_mac
        )


    def ffn_compute(self):

        return (
            self.fc1_mac
            +
            self.fc2_mac
        )


    def __repr__(self):

        return (
            f"{self.name}\n"
            f"Attention MAC:"
            f"{self.attention_compute()}\n"
            f"FFN MAC:"
            f"{self.ffn_compute()}\n"
            f"Activation:"
            f"{self.activation_size} Bytes"
        )