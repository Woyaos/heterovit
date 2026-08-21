import onnx

model = onnx.load("vit.onnx")

nodes = model.graph.node


count = {}

for node in nodes:
    op = node.op_type
    count[op] = count.get(op,0)+1


for k,v in count.items():
    print(k,v)