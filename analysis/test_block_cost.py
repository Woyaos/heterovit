from analysis.build_blocks import blocks
from runtime.cost_model.block_cost import BlockCostModel



cost=BlockCostModel()


for b in blocks:


    result=cost.estimate(b)


    print("================")

    print(b.name)


    for k,v in result.items():

        print(
            k,
            ":",
            v,
            "s"
        )