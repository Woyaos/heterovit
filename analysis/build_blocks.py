from runtime.block import BlockOperator



def build_blocks(model):

    """
    根据ViT模型构建BlockOperator列表

    输入:
        model: timm ViT模型

    输出:
        blocks: List[BlockOperator]
    """


    blocks = []


    for i, block in enumerate(model.blocks):


        op = BlockOperator(
            f"Block{i}"
        )


        # ======================
        # Attention部分
        # ======================

        # qkv Linear
        op.qkv_mac = (
            block.attn.qkv.in_features
            *
            block.attn.qkv.out_features
        )


        # attention projection
        op.proj_mac = (
            block.attn.proj.in_features
            *
            block.attn.proj.out_features
        )


        # 注意：
        # 这里暂时没有计算QK^T和Softmax
        # 后续cost model再补充
        op.attention_mac = 0



        # ======================
        # FFN部分
        # ======================

        # fc1:
        # 384 -> 1536

        op.fc1_mac = (
            block.mlp.fc1.in_features
            *
            block.mlp.fc1.out_features
        )


        # fc2:
        # 1536 -> 384

        op.fc2_mac = (
            block.mlp.fc2.in_features
            *
            block.mlp.fc2.out_features
        )


        blocks.append(op)



    return blocks





# ==========================
# 单独运行测试
# ==========================

if __name__ == "__main__":


    from models.vit import get_model


    model = get_model()


    blocks = build_blocks(model)



    for b in blocks:

        print(b)