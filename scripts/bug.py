from typing import OrderedDict
import numpy as np
import torch as th
import torch.nn as nn
import torch.onnx
import onnxruntime
import onnx

class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self, hidden_size: int, num_hidden_layers: int, output_size: int):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size

        self.initialized = False

    def _initialize(self, inputs : th.Tensor):
        if not self.initialized:
            input_size = inputs.shape[1]

            l = OrderedDict()
            l['input'] = nn.Linear(input_size, self.hidden_size)
            l['relu_in'] = nn.ReLU()
            for i in range(self.num_hidden_layers):
                l['h%d' % i] = nn.Linear(self.hidden_size, self.hidden_size)
                l['relu%d' % i] = nn.ReLU()
            l['out'] = nn.Linear(self.hidden_size, self.output_size)

            self.layers = nn.Sequential(l)
            self.initialized = True

    def forward(self, x):
        self._initialize(x)

        return self.layers(x)


class GraphIndependentModule(nn.Module):
    def __init__(self, node_model):
        super(GraphIndependentModule, self).__init__()
        self.node_model = node_model

    def forward(self, x : th.Tensor):
        x = self.node_model(x)
        return x

class GraphNetwork(nn.Module):
    def __init__(self):
        super(GraphNetwork, self).__init__()

        node_encode_model = th.nn.Sequential( MLP(128, 2, 128), th.nn.LayerNorm(128) )
        self.encoder_network = GraphIndependentModule(node_encode_model)
        self.decoder_network = MLP(128, 2, 3)

    def forward(self, x: th.Tensor) -> th.Tensor:
        node_feats = x.clone().detach()

        node_feats = self.encoder_network(node_feats)

        return self.decoder_network(node_feats)
        

if __name__ == "__main__":

    num_edges = np.random.randint(1000, 5000)
    num_nodes = 300

    edge_attr = th.rand(num_edges, 1)

    x = th.rand(num_nodes, 9)

    model = GraphNetwork()
    model(x)
    input_values = (x)
    input_names = ['node_attr']

    np.set_printoptions(threshold=6)

    model.eval()
    result = model(x).detach().numpy()
    
    torch.onnx.export(model, input_values, "H:\\Animating Tools\\Projects\\Houdini\\LearningPhysics\\scripts\\test_model.onnx", opset_version=16, input_names=input_names,
                        output_names=['coords'], dynamic_axes={'node_attr':{0:'num_nodes'}}, verbose=False, export_params=True, do_constant_folding=False)

    ort_session = onnxruntime.InferenceSession('test_model.onnx')

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    ort_inputs = { ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    output = ort_outs[0]

    print(result)
    print('----')

    print(output)

    print('-----')
    print(f'Difference is {output - result}')

