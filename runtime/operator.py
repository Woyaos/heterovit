class LinearOperator:


    def __init__(
        self,
        name,
        in_features,
        out_features
    ):

        self.name = name

        self.in_features = in_features

        self.out_features = out_features


        # MAC计算量
        self.mac = (
            in_features *
            out_features
        )


        # activation大小
        # FP16:
        # 一个元素2 Bytes

        self.input_size = (
            in_features *
            2
        )


        self.output_size = (
            out_features *
            2
        )


        # 当前通信模型使用
        self.activation_size = (
            self.input_size
            +
            self.output_size
        )