import torch
import torch.onnx
import helpers as h
import argparse
import os
from components import Normalize


def remove_normalize(model):
    # Hardcoded right now to just remove the Normalize layer and wrappers
    first_seq_children = list(list(model.children())[0].children())
    if isinstance(first_seq_children[0], Normalize):
        print("Removed Normalize layer")
        return first_seq_children[1]
        #return model
    else:
        return model
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DiffAI To ONNX Converter')
    parser.add_argument('pynet_file', type=str, help='Saved PyTorch model to convert to ONNX')

    args = parser.parse_args()

    # Hardcoded for MNIST dataset
    input_dims = torch.Size([1, 28, 28])

    model_net = torch.load(args.pynet_file)
    model_net.remove_norm()
    model_net = remove_normalize(model_net)

    onnx_file = os.path.splitext(args.pynet_file)[0] + ".onnx"
    torch.onnx.export(model_net, h.zeros([1] + list(input_dims)), onnx_file, verbose=False,
                    input_names=["actual_input"] + ["param"+str(i) for i in range(len(list(model_net.parameters())))],
                    output_names=["output"])
