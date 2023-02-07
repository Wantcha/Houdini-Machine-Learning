from typing import OrderedDict
from unicodedata import normalize
import numpy as np
import torch as th
import torch.nn as nn
#import torch.onnx
from torch.nn import Embedding, MSELoss
from torch_geometric.utils import scatter
import torch.optim as optim
#from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import torch_geometric.data as tgd
from torch_geometric.nn import MessagePassing
import collections
import os
import geo_loader as gl
from pathlib import Path
import matplotlib.pyplot as plt
#from sklearn.model_selection import learning_curve
import time


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

Stats = collections.namedtuple("Stats", ["mean", "std"])

VELOCITY_STATS = Stats(
    mean=np.zeros([3], dtype=np.float32),
    std=np.ones([3], dtype=np.float32))
ACCELERATION_STATS = Stats(
    mean=np.zeros([3], dtype=np.float32),
    std=np.ones([3], dtype=np.float32))


class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self, hidden_size: int, num_hidden_layers: int, output_size: int):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size
        self.layers = nn.Sequential()

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
            #print("INITIALIZED MLP")

    def forward(self, x):
        self._initialize(x)

        return self.layers(x)
        

def build_mlp_with_layer_norm(hidden_size: int, num_hidden_layers: int, output_size: int) -> th.nn.Module:
    mlp = MLP(hidden_size, num_hidden_layers, output_size)
    #return mlp
    return th.nn.Sequential( mlp, th.nn.LayerNorm(output_size) )


class InteractionNetworkModule(MessagePassing):
    def __init__(self, node_model, edge_model):
        super(InteractionNetworkModule, self).__init__(aggr = 'add')
        self.node_model = node_model
        self.edge_model = edge_model

    def forward(self, x : th.Tensor, edge_index: th.Tensor, edge_feats: th.Tensor):
        # EDGE BLOCK (note: add globals to them once you have them)
        #edgeFeats = []
        #edgeFeats.append(g.edge_attr)

        #edgeFeats.append( g.x[g.edge_index[0]] ) # sender data
        #print(g.x.shape)
        #print(g.x[g.edge_index[0]].shape)
        #print(g.edge_index.shape)
        # note: should we also append receiver data?

        # src, dst = edge_index
        collectedEdgeFeats = th.concat([edge_feats, x[edge_index[0]]], dim=1) # send and receiver data is the same (bidirectional graph)

        new_edge_feats = self.edge_model(collectedEdgeFeats)

        new_node_feats = scatter(new_edge_feats, edge_index[0], dim=0, dim_size=x.size(0))
        #print("Node Feats")
        #print(new_node_feats)
        collected_node_feats = th.concat([x, new_node_feats], dim=1)
        out_nodes = self.node_model(collected_node_feats)

        #out_nodes = self.propagate(edge_index=edge_index, size=(x.size(0), x.size(0)), new_edge_feats=new_edge_feats, x=x)

        return out_nodes, new_edge_feats

    def message(self, edge_index_j: th.Tensor, new_edge_feats: th.Tensor, edge_index, x_j) -> th.Tensor:
        # return super().message(x_j)
        print(edge_index)
        print(edge_index_j)
        return new_edge_feats[edge_index_j]

    def update(self, aggr_out: th.Tensor, x: th.Tensor) -> th.Tensor:
        # graph['edge_attr'] = g.edge_attr + new_edge_feats
        collected_node_feats = th.concat([x, aggr_out], dim=1)

        return self.node_model(collected_node_feats)
        #return super().update(inputs)



class GraphIndependentModule(nn.Module):
    def __init__(self, node_model, edge_model):
        super(GraphIndependentModule, self).__init__()
        self.node_model = node_model
        self.edge_model = edge_model

    def forward(self, x : th.Tensor, edge_feats: th.Tensor):
        # g.ndata['EncodedFeats'] = self.node_model(g.ndata['Feats'])
        # g.edata['EncodedFeats'] = self.edge_model(g.edata['Feats'])

        x = self.node_model(x)
        edge_feats = self.edge_model(edge_feats)
        return x, edge_feats
    
NETWORK_GLOBALS = { "mlp_hidden_size": 128,
                    "mlp_num_hidden_layers": 2,
                    "mlp_latent_size": 128,
                    "message_passing_steps": 10 }

class GraphNetwork(nn.Module):
    def __init__(self, particle_embed_size):
        super(GraphNetwork, self).__init__()
        self.embedParticles = nn.Embedding(2, particle_embed_size) # nn.Embedding(len(np.unique(particle_types)), particle_embed_size)

        mlp_hidden_size = NETWORK_GLOBALS["mlp_hidden_size"]
        mlp_num_hidden_layers = NETWORK_GLOBALS["mlp_num_hidden_layers"]
        mlp_latent_size = NETWORK_GLOBALS["mlp_latent_size"]
        message_passing_steps = NETWORK_GLOBALS["message_passing_steps"]

        edge_encode_model = build_mlp_with_layer_norm(mlp_hidden_size, mlp_num_hidden_layers, mlp_latent_size)
        node_encode_model = build_mlp_with_layer_norm(mlp_hidden_size, mlp_num_hidden_layers, mlp_latent_size)
        self.encoder_network = GraphIndependentModule(node_encode_model, edge_encode_model)

        self.processor_networks = nn.ModuleList()
        for _ in range(message_passing_steps):
            self.processor_networks.append(
                InteractionNetworkModule(build_mlp_with_layer_norm(mlp_hidden_size, mlp_num_hidden_layers, mlp_latent_size),
                                            build_mlp_with_layer_norm(mlp_hidden_size, mlp_num_hidden_layers, mlp_latent_size)))

        self.decoder_network = MLP(mlp_hidden_size, mlp_num_hidden_layers, 3)

    def forward(self, x: th.Tensor, edge_index: th.Tensor, edge_attr : th.Tensor, particle_types : th.Tensor) -> th.Tensor:
        particle_type_embeddings = self.embedParticles(particle_types)

        node_feats = th.cat((x, particle_type_embeddings), 1)
        #print(node_feats.shape)

        node_feats, edge_feats = self.encoder_network(node_feats, edge_attr)

        for processor_network in self.processor_networks:
            # g['x'] = g.x + newNodes
            # g['edge_attr'] = g.edge_attr + newEdges
            processed_node_feats, processed_edge_feats = processor_network(node_feats, edge_index, edge_feats)
            node_feats = node_feats + processed_node_feats
            edge_feats = edge_feats + processed_edge_feats

        return self.decoder_network(node_feats)

class HoudiniFileSequence():
    def __init__(self, base_name: str, index: int, sequence_length: int):
        self.base_name = base_name
        self.index = index
        self.sequence_length = sequence_length

class SimulationsDataset(Dataset):
    def __init__(self, file_names: list):
        super().__init__()
        self.file_names = file_names
        #self.filepath = filepath

    def len(self):
        return len(self.file_names)

    def get(self, index):

        g = th.load(self.file_names[index])
        non_kinematic_mask = th.logical_not(g.pinned_points)

        sampled_noise = gl.get_random_walk_noise_for_position_sequence(g.original_inputs, noise_std)
        target_shape = sampled_noise.shape

        noise_mask = non_kinematic_mask.view(len(non_kinematic_mask), *(1,)*(len(target_shape) - 1)).expand(target_shape)
        sampled_noise *= noise_mask.float()

        g.x = gl.calculate_noisy_velocities(g.original_inputs.permute(1,0,2), sampled_noise.permute(1,0,2))

        g.last_position_noise = sampled_noise[:,-1,:]

        '''
        cur_sequence: HoudiniFileSequence = self.file_names[index]
        #print(self.filepath, cur_sequence.base_name)
        geometry_sequence: gl.GeometrySequence = gl.load_geometry_sequence(self.filepath, cur_sequence.base_name,
                                                                            cur_sequence.index, cur_sequence.sequence_length)

        g = gl.encodePreprocess(geometry_sequence.input_positions_sequence, geometry_sequence.mesh)
        g.targets = geometry_sequence.target_positions
        g.pinned_points = geometry_sequence.pinned_points
        g.original_inputs = geometry_sequence.input_positions_sequence.permute(1, 0, 2) # for batching

        #print(g.targets.shape)
        #print(g.pinned_points.shape)
        #print(g.original_inputs.shape)
        '''

        return g


def output_postprocessing(input_positions_sequence, normalized_acceleration):
    predicted_acceleration = (normalized_acceleration * ACCELERATION_STATS.std) + ACCELERATION_STATS.mean
    most_recent_position = input_positions_sequence[-1]
    most_recent_velocity = most_recent_position - input_positions_sequence[-2]

    new_velocity = most_recent_velocity + predicted_acceleration
    new_position = most_recent_position + new_velocity

    #position_sequence_noise = gl.get_random_walk_noise_for_position_sequence(input_positions_sequence, noise_std_last_step=6.7e-4)

    return new_position



def get_target_normalized_acceleration(target_positions: th.Tensor, position_sequence, last_input_noise):
    #print(last_input_noise.shape)
    #print(target_positions.shape)
    next_position_adjusted = target_positions + last_input_noise

    previous_position = position_sequence[-1]
    previous_velocity = previous_position - position_sequence[-2]

    next_velocity = next_position_adjusted - previous_position
    acceleration = next_velocity - previous_velocity

    normalized_acceleration = (acceleration - gl.ACCELERATION_STATS.mean) / gl.ACCELERATION_STATS.std

    return normalized_acceleration


noise_std = 6.7e-6
particle_type_embedding_size = 8
model_path = "H:\\Animating Tools\\Projects\\Houdini\\LearningPhysics\\scripts\\physics_model.pt"

def draw_learning_curve(average_losses: list, dir_path: str, batch_size, train_size):
    epoch_nums = list(range(1, len(average_losses) + 1))

    lines = plt.plot(epoch_nums, average_losses)
    plt.setp(lines, linewidth=2.0)
    plt.text(3, (average_losses[0] + average_losses[len(average_losses) - 1]) * 0.6, f"Batch Size: {batch_size}, Sequences: {train_size}\n"
                        f"MLP Size: {NETWORK_GLOBALS['mlp_hidden_size']}, Message Steps: {NETWORK_GLOBALS['message_passing_steps']}")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")

    filename = f"batch{batch_size}_seq{train_size}_hiddensize{NETWORK_GLOBALS['mlp_hidden_size']}_messages{NETWORK_GLOBALS['message_passing_steps']}.png"
    plt.savefig(Path(dir_path, filename))
    #plt.show()


if __name__ == "__main__":
    print('Loading')
    geoPath = 'H:\Animating Tools\Projects\Houdini\LearningPhysics\geo'
    plot_path = 'H:\Animating Tools\Projects\Houdini\LearningPhysics\learning_plots\\acceleration'
    preprocessed_dataset_path = 'H:\Animating Tools\Projects\Houdini\LearningPhysics\processed_geo\cloth'

    sequence_length = 6
    learning_rate = 1e-4
    min_learning_rate = 1e-7
    batch_size = 4
    num_epochs = 25
    decay_rate = 0.9

    reprocess_dataset = False

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    file_sequences = []
    for root, dirs, files in os.walk(geoPath):
        for dir in dirs:
            files = os.listdir(Path(root, dir))
            base_name = files[0].split('.', 1)[0]
            for i in range(1, len(files) - sequence_length + 2):
                seq = HoudiniFileSequence(Path(dir, base_name), i, sequence_length)
                file_sequences.append(seq)

    print(f"We got {len(file_sequences)} sequences to work with")

    if reprocess_dataset:
        print("Re-processing dataset...")
        for filename in os.listdir(preprocessed_dataset_path):
            file_path = os.path.join(preprocessed_dataset_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        i = 0
        for seq in file_sequences:
            geometry_sequence: gl.GeometrySequence = gl.load_geometry_sequence(geoPath, seq.base_name,
                                                                                seq.index, seq.sequence_length)

            g = gl.encodePreprocess(geometry_sequence.input_positions_sequence, geometry_sequence.mesh)
            g.targets = geometry_sequence.target_positions
            g.pinned_points = geometry_sequence.pinned_points
            g.original_inputs = geometry_sequence.input_positions_sequence.permute(1, 0, 2)

            th.save(g, Path(preprocessed_dataset_path, f"{i}.txt"))
            i += 1
            print(i)


    dataset = SimulationsDataset([Path(preprocessed_dataset_path, f"{index}.txt") for index in range(len(file_sequences))])
    #dataset = SimulationsDataset(file_sequences, geoPath)
    train_test_split = 0.8
    train_dataset = dataset[:int(len(dataset) * train_test_split)]
    test_dataset = dataset[int(len(dataset) * train_test_split):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GraphNetwork(particle_type_embedding_size)

    # Training
    model.train()

    criterion = MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    lr_lambda = lambda epoch: decay_rate ** epoch if decay_rate ** epoch >= min_learning_rate else min_learning_rate
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda, verbose=True)

    avg_losses = []
    for epoch in range(num_epochs):
        losses = []
        print(f"Epoch {epoch + 1}")
        start_time = time.time()

        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()

            predicted_normalized_acceleration: th.Tensor = model(data.x, data.edge_index, data.edge_attr, data.pinned_points)

            targets = data.targets.clone()
            data.original_inputs = data.original_inputs.permute(1, 0, 2)

            accel_targets = get_target_normalized_acceleration(targets, data.original_inputs, data.last_position_noise)
            
            accel_targets = accel_targets.to(device=device)
            predicted_normalized_acceleration = predicted_normalized_acceleration.to(device=device)


            loss: th.Tensor = criterion(predicted_normalized_acceleration, accel_targets)

            loss_mask = data.pinned_points.view(len(data.pinned_points), *(1,)*(len(loss.shape) - 1)).expand(loss.shape).to(device=device)
            loss = th.sum(th.where(loss_mask == 0, loss, th.zeros(loss.shape).to(device=device)))

            num_non_pinned = data.pinned_points.size(0) - th.sum(data.pinned_points)
            
            loss = loss / num_non_pinned
            #print(loss)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        scheduler.step()

        avg_loss = 0
        for l in losses:
            avg_loss += l
        avg_loss = avg_loss / len(losses)
        print(f"Average loss is {avg_loss:.4f}")
        avg_losses.append(avg_loss)
        print(f"Epoch took {(time.time() - start_time):.2f} seconds")

    draw_learning_curve(avg_losses, plot_path, batch_size, len(train_dataset))

    
    test_criterion = MSELoss()
    #Testing
    model.eval()
    print("Testing")
    for batch_idx, data in enumerate(test_loader):
        data.original_inputs = data.original_inputs.permute(1, 0, 2)

        predicted_normalized_acceleration: th.Tensor = model(data.x, data.edge_index, data.edge_attr, data.pinned_points).detach().numpy()
        predicted_normalized_acceleration = predicted_normalized_acceleration.reshape(
                                                    (int(predicted_normalized_acceleration.shape[0] / data.original_inputs.shape[1]),
                                                    data.original_inputs.shape[1], -1))

        #print(predicted_normalized_acceleration.shape)
        #print(data.original_inputs.shape)

        predicted_positions = output_postprocessing(data.original_inputs, predicted_normalized_acceleration)
        #print(predicted_positions.shape)

        predicted_positions = predicted_positions.contiguous().view(predicted_positions.shape[1], -1)

        loss = test_criterion(predicted_positions, data.targets)
        print(f"Loss is {loss:.5f}")

    #input_values = (g.x, g.edge_index, g.edge_attr, geometry_sequence.pinned_points)
    #input_names = ['node_attr', 'edge_index', 'edge_attr', 'particle_types']

    #predicted_normalized_acceleration = model(g.x, g.edge_index, g.edge_attr, geometry_sequence.pinned_points).detach().numpy()
    #print(predicted_normalized_acceleration)

    #torch.onnx.export(model, input_values, "H:\\Animating Tools\\Projects\\Houdini\\LearningPhysics\\scripts\\physics_model.onnx", opset_version=16, input_names=input_names,
                        #output_names=['coords'], dynamic_axes={'node_attr':{0:'num_nodes'}, 'edge_index':{1:'num_edges'}, 'edge_attr':{0:'num_edges'}, 'particle_types':{0:'num_nodes'}}, verbose=False)

    th.save(model.state_dict(), model_path)

    print('done')

    # TODO: switch workflow to from standard to NOISY input

