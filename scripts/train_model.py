from typing import OrderedDict
from unicodedata import normalize
import numpy as np
import torch as th
import torch.nn as nn
#import torch.onnx
from torch.nn import Embedding, MSELoss
from torch_geometric.utils import scatter
import torch_scatter
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

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

'''VELOCITY_STATS = Stats(
    mean = th.zeros([3]).to(device),
    std = th.ones([3]).to(device))'''
ACCELERATION_STATS = Stats(
    mean = th.zeros([3]),
    std = th.ones([3]))


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

            layers = []
            layers.append( nn.Linear(input_size, self.hidden_size, device=device) )
            layers.append( nn.ReLU() )
            for i in range(self.num_hidden_layers):
                layers.append( nn.Linear(self.hidden_size, self.hidden_size, device=device) )
                layers.append( nn.ReLU() )
            layers.append( nn.Linear(self.hidden_size, self.output_size, device=device) )

            self.layers = nn.Sequential(*layers)
            self.initialized = True
            #print("INITIALIZED MLP")

    def forward(self, x):
        self._initialize(x)
        #x = x.to(device)
        output = self.layers(x)
        return output
        

def build_mlp_with_layer_norm(hidden_size: int, num_hidden_layers: int, output_size: int) -> th.nn.Module:
    mlp = MLP(hidden_size, num_hidden_layers, output_size)
    #return mlp
    return th.nn.Sequential( mlp, th.nn.LayerNorm(output_size) )


class OldInteractionNetworkModule(MessagePassing):
    def __init__(self, node_model, edge_model):
        super(OldInteractionNetworkModule, self).__init__(aggr = 'add')
        self.node_model = node_model
        self.edge_model = edge_model

    def forward(self, x : th.Tensor, edge_index: th.Tensor, edge_feats: th.Tensor):
        # EDGE BLOCK (note: add globals to them once you have them)

        # src, dst = edge_index
        collectedEdgeFeats = th.concat([edge_feats, x[edge_index[0]]], dim=1) # send and receiver data is the same (bidirectional graph)

        new_edge_feats = self.edge_model(collectedEdgeFeats)

        new_node_feats = scatter(new_edge_feats, edge_index[0], dim=0, dim_size=x.size(0))
        #print("Node Feats")
        #print(new_node_feats)
        collected_node_feats = th.concat([x, new_node_feats], dim=1)
        out_nodes = self.node_model(collected_node_feats)

        return out_nodes, new_edge_feats

    def message(self, edge_index_j: th.Tensor, new_edge_feats: th.Tensor, edge_index, x_j) -> th.Tensor:
        # return super().message(x_j)
        return new_edge_feats[edge_index_j]

    def update(self, aggr_out: th.Tensor, x: th.Tensor) -> th.Tensor:
        collected_node_feats = th.concat([x, aggr_out], dim=1)

        return self.node_model(collected_node_feats)


class InteractionNetworkModule(MessagePassing):
    def __init__(self, node_model, edge_model):
        super(InteractionNetworkModule, self).__init__(aggr = 'add')
        self.node_model = node_model
        self.edge_model = edge_model

    def forward(self, x, edge_index, e_features):
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        x_residual = x
        e_features_residual = e_features
        x, e_features = self.propagate(edge_index=edge_index, x=x, e_features=e_features)
        return x+x_residual, e_features+e_features_residual

    def message(self, edge_index, x_i, x_j, e_features):
        e_features = th.cat([x_i, x_j, e_features], dim=-1)
        e_features = self.edge_model(e_features)
        return e_features

    def update(self, x_updated, x, e_features):
        # x_updated: (E, edge_out)
        # x: (E, node_in)
        x_updated = th.cat([x_updated, x], dim=-1)
        x_updated = self.node_model(x_updated)
        return x_updated, e_features
    

class GraphNetBlock(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, node_model, mesh_edge_model, world_edge_model, message_passing_aggregator):
        super().__init__()
        self.mesh_edge_model = mesh_edge_model
        self.world_edge_model = world_edge_model
        self.node_model = node_model
        self.message_passing_aggregator = message_passing_aggregator

    def _update_edge_features(self, node_features, edge_set):
        """Aggregrates node features, and applies edge function."""
        senders = edge_set.senders
        receivers = edge_set.receivers
        sender_features = th.index_select(input=node_features, dim=0, index=senders)
        receiver_features = th.index_select(input=node_features, dim=0, index=receivers)
        features = [sender_features, receiver_features, edge_set.features]
        features = th.cat(features, dim=-1)

        if edge_set.name == "mesh_edges":
            return self.mesh_edge_model(features)
        else:
            return self.world_edge_model(features)

    def unsorted_segment_operation(self, data, segment_ids, num_segments, operation):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.
        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        data = data.to(device)
        segment_ids = segment_ids.to(device)
        if len(segment_ids.shape) == 1:
            s = th.prod(th.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:]).to(device)

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        #shape = [num_segments] + list(data.shape[1:])
        #result = th.zeros(*shape).to(device)
        if operation == 'sum':
            result = torch_scatter.scatter_add(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'max':
            result, _ = torch_scatter.scatter_max(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'mean':
            result = torch_scatter.scatter_mean(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'min':
            result, _ = torch_scatter.scatter_min(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'std':
            result = torch_scatter.scatter_std(data.float(), segment_ids, out=result, dim=0, dim_size=num_segments)
        else:
            raise Exception('Invalid operation type!')
        result = result.type(data.dtype)
        return result

    def _update_node_features(self, node_features, edge_sets):
        """Aggregrates edge features, and applies node function."""
        num_nodes = node_features.shape[0]
        features = [node_features]
        for edge_set in edge_sets:
            features.append(
                self.unsorted_segment_operation(edge_set.features, edge_set.receivers, num_nodes,
                                                operation=self.message_passing_aggregator))
        features = th.cat(features, dim=-1)
        return self.node_model(features)

    def forward(self, graph: gl.MultiGraph, mask=None):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        # apply edge functions
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            updated_features = self._update_edge_features(graph.node_features, edge_set)
            new_edge_sets.append(edge_set._replace(features=updated_features))

        # apply node function
        new_node_features = self._update_node_features(graph.node_features, new_edge_sets)

        # add residual connections
        new_node_features += graph.node_features
        if mask is not None:
            mask = mask.repeat(new_node_features.shape[-1])
            mask = mask.view(new_node_features.shape[0], new_node_features.shape[1])
            new_node_features = th.where(mask, new_node_features, graph.node_features)

        new_edge_sets = [es._replace(features=es.features + old_es.features)
                         for es, old_es in zip(new_edge_sets, graph.edge_sets)]
        return gl.MultiGraph(new_node_features, new_edge_sets)


class MultiGraphIndependentModule(nn.Module):
    def __init__(self, node_model, mesh_edge_model, world_edge_model):
        super(MultiGraphIndependentModule, self).__init__()
        self.node_model = node_model
        self.mesh_edge_model = mesh_edge_model
        self.world_edge_model = world_edge_model

    def forward(self,graph: gl.MultiGraph):
        node_latents = self.node_model(graph.node_features)
        new_edges_sets = []

        #print(graph.node_features.shape)
        #print(len(graph.edge_sets))
        for index, edge_set in enumerate(graph.edge_sets):
            #print(len(edge_set))
            if edge_set.name == "mesh_edges":
                feature = edge_set.features
                latent = self.mesh_edge_model(feature)
                new_edges_sets.append(edge_set._replace(features=latent))
            else:
                feature = edge_set.features
                latent = self.world_edge_model(feature)
                new_edges_sets.append(edge_set._replace(features=latent))
        return gl.MultiGraph(node_latents, new_edges_sets)
    
    
NETWORK_GLOBALS = { "mlp_hidden_size": 128,
                    "mlp_num_hidden_layers": 2,
                    "mlp_latent_size": 128,
                    "message_passing_steps": 15 }

class GraphNetwork(nn.Module):
    def __init__(self):
        super(GraphNetwork, self).__init__()
        #self.embedParticles = nn.Embedding(2, particle_embed_size) # nn.Embedding(len(np.unique(particle_types)), particle_embed_size)

        mlp_hidden_size = NETWORK_GLOBALS["mlp_hidden_size"]
        mlp_num_hidden_layers = NETWORK_GLOBALS["mlp_num_hidden_layers"]
        mlp_latent_size = NETWORK_GLOBALS["mlp_latent_size"]
        message_passing_steps = NETWORK_GLOBALS["message_passing_steps"]

        mesh_edge_encode_model = build_mlp_with_layer_norm(mlp_hidden_size, mlp_num_hidden_layers, mlp_latent_size)
        world_edge_encode_model = build_mlp_with_layer_norm(mlp_hidden_size, mlp_num_hidden_layers, mlp_latent_size)
        node_encode_model = build_mlp_with_layer_norm(mlp_hidden_size, mlp_num_hidden_layers, mlp_latent_size)
        self.encoder_network = MultiGraphIndependentModule(node_encode_model, mesh_edge_encode_model, world_edge_encode_model)

        self.processor_networks = nn.ModuleList()
        for _ in range(message_passing_steps):
            self.processor_networks.append(
                GraphNetBlock(build_mlp_with_layer_norm(mlp_hidden_size, mlp_num_hidden_layers, mlp_latent_size),
                              build_mlp_with_layer_norm(mlp_hidden_size, mlp_num_hidden_layers, mlp_latent_size),
                              build_mlp_with_layer_norm(mlp_hidden_size, mlp_num_hidden_layers, mlp_latent_size), "sum"))

        self.decoder_network = MLP(mlp_hidden_size, mlp_num_hidden_layers, 3)

    def forward(self, multigraph: gl.MultiGraph) -> th.Tensor:
        #particle_type_embeddings = self.embedParticles(particle_types).to(device)

        #node_feats = th.cat((x, particle_type_embeddings), 1).to(device)
        #print(node_feats.shape)

        multigraph = self.encoder_network(multigraph)

        for processor_network in self.processor_networks:
            multigraph = processor_network(multigraph)

        node_feats = self.decoder_network(multigraph.node_features)
        return node_feats


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

        #non_kinematic_mask = th.logical_not(g.pinned_points).clone().detach().to(device)

        sampled_noise = gl.get_random_walk_noise_for_position_sequence(g["point"].original_inputs, noise_std)#.to(device)

        #sampled_noise *= non_kinematic_mask.float().view(-1, 1, 1)

        #g.x = gl.calculate_noisy_velocities(g.original_inputs.to(device), sampled_noise)

        g['point'].last_position_noise = sampled_noise[:,-1]

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
    most_recent_position = input_positions_sequence[:, -1]
    #print(most_recent_position.shape)
    most_recent_velocity = most_recent_position - input_positions_sequence[:, -2]

    new_velocity = most_recent_velocity + predicted_acceleration
    new_position = most_recent_position + new_velocity

    #position_sequence_noise = gl.get_random_walk_noise_for_position_sequence(input_positions_sequence, noise_std_last_step=6.7e-4)

    return new_position



def get_target_normalized_acceleration(target_positions: th.Tensor, position_sequence, last_input_noise):
    #print(last_input_noise.shape)
    #print(target_positions.shape)
    next_position_adjusted = target_positions# + last_input_noise

    previous_position = position_sequence[:, -1]
    previous_velocity = previous_position - position_sequence[:, -2]

    next_velocity = next_position_adjusted - previous_position
    acceleration = next_velocity - previous_velocity

    normalized_acceleration = (acceleration - gl.ACCELERATION_STATS.mean) / gl.ACCELERATION_STATS.std

    return normalized_acceleration


noise_std = 6.7e-6
#particle_type_embedding_size = 8
model_path = "H:\\Animating Tools\\Projects\\Houdini\\LearningPhysics\\scripts\\physics_model.pt"

def draw_learning_curve(average_losses: list, dir_path: str, batch_size, train_size, epochs):
    epoch_nums = list(range(1, len(average_losses) + 1))

    lines = plt.plot(epoch_nums, average_losses)
    plt.setp(lines, linewidth=2.0)
    plt.text(epochs/3, (average_losses[0] + average_losses[len(average_losses) - 1]) * 0.5, f"Batch Size: {batch_size}, Sequences: {train_size}\n"
                        f"MLP Size: {NETWORK_GLOBALS['mlp_hidden_size']}, Message Steps: {NETWORK_GLOBALS['message_passing_steps']}")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")

    filename = f"batch{batch_size}_seq{train_size}_hiddensize{NETWORK_GLOBALS['mlp_hidden_size']}_messages{NETWORK_GLOBALS['message_passing_steps']}_epochs{epochs}.png"
    plt.savefig(Path(dir_path, filename))
    #plt.show()


if __name__ == "__main__":
    print('Loading')
    geoPath = 'H:\Animating Tools\Projects\Houdini\LearningPhysics\geo'
    plot_path = 'H:\Animating Tools\Projects\Houdini\LearningPhysics\learning_plots\\acceleration'
    preprocessed_dataset_path = 'H:\Animating Tools\Projects\Houdini\LearningPhysics\processed_geo\cloth'

    sequence_length = 3
    learning_rate = 1e-4
    min_learning_rate = 1e-8
    batch_size = 2
    num_epochs = 25
    decay_rate = 0.85

    reprocess_dataset = False
    delete_old_files = True

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
        if delete_old_files:
            for filename in os.listdir(preprocessed_dataset_path):
                file_path = os.path.join(preprocessed_dataset_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        for i, seq in enumerate(file_sequences):
            geometry_sequence: gl.GeometrySequence = gl.load_geometry_sequence(geoPath, seq.base_name,
                                                                                seq.index, seq.sequence_length)
            g = gl.encodePreprocess(geometry_sequence.input_positions_sequence, geometry_sequence.mesh, geometry_sequence.pinned_points)

            g["point"].targets = geometry_sequence.target_positions
            g["point"].pinned_points = geometry_sequence.pinned_points
            g["point"].original_inputs = geometry_sequence.input_positions_sequence
            #print(f"{g.targets.shape}, {g.pinned_points.shape}, {g.original_inputs.shape}")

            th.save(g, Path(preprocessed_dataset_path, f"{i}.txt"))
            if i % 500 == 0:
                print(f"{int(i / len(file_sequences) * 100)}%")


    dataset = SimulationsDataset([Path(preprocessed_dataset_path, f"{index}.txt") for index in range(len(file_sequences))])
    #dataset = SimulationsDataset(file_sequences, geoPath)
    train_test_split = 0.8
    train_dataset = dataset[:int(len(dataset) * train_test_split)]
    test_dataset = dataset[int(len(dataset) * train_test_split):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GraphNetwork().to(device=device)

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

            mesh_edges = gl.EdgeSet(
                    name='mesh_edges',
                    features=data["point", "mesh", "point"].edge_attr.to(device),
                    receivers=data["point", "mesh", "point"].edge_index[0].to(device),
                    senders=data["point", "mesh", "point"].edge_index[1].to(device))

            world_edges = gl.EdgeSet(
                    name='world_edges',
                    features=data["point", "world", "point"].edge_attr.to(device),
                    receivers=data["point", "world", "point"].edge_index[0].to(device),
                    senders=data["point", "world", "point"].edge_index[1].to(device))
            

            multigraph = gl.MultiGraph(node_features=data["point"].x.to(device), edge_sets=[mesh_edges, world_edges])

            predicted_normalized_acceleration: th.Tensor = model(multigraph)

            accel_targets = get_target_normalized_acceleration(data["point"].targets, data["point"].original_inputs, data["point"].last_position_noise).to(device)

            loss: th.Tensor = criterion(predicted_normalized_acceleration, accel_targets)

            loss = th.sum(loss, dim=-1)

            loss_mask = data["point"].pinned_points.to(device)
            loss = th.sum(th.where(loss_mask == 0, loss, th.zeros(loss.shape).to(device)))

            num_non_pinned = data["point"].pinned_points.size(0) - th.sum(data["point"].pinned_points)
            
            loss = loss / num_non_pinned

            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

            del loss_mask
            del accel_targets
            del num_non_pinned
            del predicted_normalized_acceleration

        scheduler.step()

        avg_loss = 0
        for l in losses:
            avg_loss += l
        avg_loss = avg_loss / len(losses)
        print(f"Average loss is {avg_loss:.4f}")
        avg_losses.append(avg_loss)
        print(f"Epoch took {(time.time() - start_time):.2f} seconds")

    draw_learning_curve(avg_losses, plot_path, batch_size, len(train_dataset), num_epochs)

    
    test_criterion = MSELoss()
    #Testing
    model.eval()
    print("Testing")
    test_losses_sum = 0
    num_tests = 0
    for batch_idx, data in enumerate(test_loader):
        data.targets = data.targets.to(device)

        predicted_normalized_acceleration: th.Tensor = model(data.x.to(device=device), data.edge_index.to(device=device), data.edge_attr.to(device=device), data.pinned_points.to(device=device)).to(device)

        predicted_positions = output_postprocessing(data.original_inputs.to(device=device), predicted_normalized_acceleration.to(device))
        kinematic_mask = data.pinned_points.to(device=device)
        kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, 3)

        predicted_positions = th.where(kinematic_mask, predicted_positions, data.targets)

        loss = test_criterion(predicted_positions, data.targets)
        print(f"Loss is {loss:.5f}")
        test_losses_sum += loss.item()
        num_tests += 1
    print(f"Average test loss is {(test_losses_sum / num_tests):.7f}")

    #input_values = (g.x, g.edge_index, g.edge_attr, geometry_sequence.pinned_points)
    #input_names = ['node_attr', 'edge_index', 'edge_attr', 'particle_types']

    #predicted_normalized_acceleration = model(g.x, g.edge_index, g.edge_attr, geometry_sequence.pinned_points).detach().numpy()
    #print(predicted_normalized_acceleration)

    #torch.onnx.export(model, input_values, "H:\\Animating Tools\\Projects\\Houdini\\LearningPhysics\\scripts\\physics_model.onnx", opset_version=16, input_names=input_names,
                        #output_names=['coords'], dynamic_axes={'node_attr':{0:'num_nodes'}, 'edge_index':{1:'num_edges'}, 'edge_attr':{0:'num_edges'}, 'particle_types':{0:'num_nodes'}}, verbose=False)

    th.save(model.state_dict(), model_path)

    print('done')

    # TODO: switch workflow to from standard to NOISY input

