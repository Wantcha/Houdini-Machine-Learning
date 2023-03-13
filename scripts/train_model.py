from typing import OrderedDict
from unicodedata import normalize
import torch as th
import torch.nn as nn
#import torch.onnx
#from torch_geometric.utils import scatter
import torch_scatter
import torch.optim as optim
#from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import torch_geometric.data as tgd
from torch_geometric.nn import MessagePassing
import os
import geo_loader as gl
from pathlib import Path
import matplotlib.pyplot as plt
import time

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

device = th.device('cuda' if th.cuda.is_available() else 'cpu')


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


def save_checkpoint(checkpoint_path, epoch: int, model, optimizer, scheduler, normalizers: dict, loss):
    chk_name = f"checkpoint_loss{loss:.3f}_epoch{epoch}.pt"
    th.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'output_norm': normalizers['output'].state_dict(),
            'node_norm': normalizers['node'].state_dict(),
            'mesh_edge_norm': normalizers['mesh_edge'].state_dict(),
            'world_edge_norm': normalizers['world_edge'].state_dict(),
        },
        Path(checkpoint_path, chk_name)
    )
    print("Saved checkpoint " + chk_name)

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
        #assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        data = data.to(device)
        segment_ids = segment_ids.to(device)
        if len(segment_ids.shape) == 1:
            s = th.prod(th.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:]).to(device)

        #assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

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
        '''if mask is not None:
            mask = mask.repeat(new_node_features.shape[-1])
            mask = mask.view(new_node_features.shape[0], new_node_features.shape[1])
            new_node_features = th.where(mask, new_node_features, graph.node_features)'''

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


class Normalizer(nn.Module):
    """Feature normalizer that accumulates statistics online."""

    def __init__(self, size, name, max_accumulations=10 ** 5, std_epsilon=1e-8):
        super(Normalizer, self).__init__()
        self._name = name
        self._max_accumulations = max_accumulations
        self._std_epsilon = th.tensor([std_epsilon], requires_grad=False).to(device)

        self._acc_count = th.zeros(1, dtype=th.float32, requires_grad=False).to(device)
        self._num_accumulations = th.zeros(1, dtype=th.float32, requires_grad=False).to(device)
        self._acc_sum = th.zeros(size, dtype=th.float32, requires_grad=False).to(device)
        self._acc_sum_squared = th.zeros(size, dtype=th.float32, requires_grad=False).to(device)

    def forward(self, batched_data, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        if accumulate and self._num_accumulations < self._max_accumulations:
            # stop accumulating after a million updates, to prevent accuracy issues
            self._accumulate(batched_data)
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data):
        """Function to perform the accumulation of the batch_data statistics."""
        count = th.tensor(batched_data.shape[0], dtype=th.float32, device=device)

        data_sum = th.sum(batched_data, dim=0)
        squared_data_sum = th.sum(batched_data ** 2, dim=0)
        self._acc_sum = self._acc_sum.add(data_sum)
        self._acc_sum_squared = self._acc_sum_squared.add(squared_data_sum)
        self._acc_count = self._acc_count.add(count.item())
        self._num_accumulations = self._num_accumulations.add(1.)
        #del count
        #del data_sum
        #del squared_data_sum

    def _mean(self):
        safe_count = th.maximum(self._acc_count, th.tensor([1.], device=device))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = th.maximum(self._acc_count, th.tensor([1.], device=device))
        std = th.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
        return th.maximum(std, self._std_epsilon)

    def get_acc_sum(self):
        return self._acc_sum
    
    def save(self, out_path):
        th.save({
            "name": self._name,
            "max_accumulations": self._max_accumulations,
            "std_epsilon": self._std_epsilon,
            "acc_count": self._acc_count,
            "num_accumulations": self._num_accumulations,
            "acc_sum": self._acc_sum,
            "acc_sum_squared": self._acc_sum_squared
        }, out_path)
    
    
NETWORK_GLOBALS = { "mlp_hidden_size": 128,
                    "mlp_num_hidden_layers": 2,
                    "mlp_latent_size": 128,
                    "message_passing_steps": 18 }

class GraphNetwork(nn.Module):
    def __init__(self):
        super(GraphNetwork, self).__init__()

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

    def len(self):
        return len(self.file_names)

    def get(self, index):

        #print(self.file_names[index])
        g = th.load(self.file_names[index])

        with th.no_grad():
            non_kinematic_mask = th.eq(g["point"].point_types.to(device), th.tensor([0], device=device).int())

            sampled_noise = gl.get_random_walk_noise_for_position_sequence(g["point"].original_inputs.to(device), noise_std)
            sampled_noise *= non_kinematic_mask.float().view(-1, 1, 1)

            g["point"].original_inputs = g["point"].original_inputs.to(device)
            g["point"].x = g["point"].x.to(device)
            g["point"].x = gl.calculate_noisy_node_velocity(g["point"].x, g["point"].original_inputs, sampled_noise)
            g["point"].targets = g["point"].targets.to(device) + (1.0 - noise_gamma) * sampled_noise[:, -1]

        return g


def output_postprocessing(input_positions_sequence, acceleration):
    new_position = 2 * input_positions_sequence[:, -1] + acceleration - input_positions_sequence[:, -2]
    return new_position



def get_target_normalized_acceleration(target_positions: th.Tensor, position_sequence):
    return target_positions - 2 * position_sequence[:, -1] + position_sequence[:, -2]


def draw_learning_curve(average_losses: list, dir_path: str, batch_size, train_size, epochs, lr_decay):
    epoch_nums = list(range(1, len(average_losses) + 1))

    lines = plt.plot(epoch_nums, average_losses)
    plt.rcParams.update({'font.size' : 8})
    plt.setp(lines, linewidth=2.0)
    plt.text(len(average_losses)/3.0, (average_losses[0] + average_losses[len(average_losses) - 1]) * 0.5, f"Batch Size: {batch_size}, Sequences: {train_size}\n"
                        f"MLP Size: {NETWORK_GLOBALS['mlp_hidden_size']}, MLP Depth: {NETWORK_GLOBALS['mlp_num_hidden_layers']}, Message Steps: {NETWORK_GLOBALS['message_passing_steps']}\n"
                        f"Decay Rate:{lr_decay}, Noise_Std: {noise_std}")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")

    filename = (f"batch{batch_size}_seq{train_size}_hiddensize{NETWORK_GLOBALS['mlp_hidden_size']}"
                f"_hiddenwidth{NETWORK_GLOBALS['mlp_num_hidden_layers']}_messages{NETWORK_GLOBALS['message_passing_steps']}_epochs{epochs}_decay{lr_decay}.png")
    plt.savefig(Path(dir_path, filename))
    plt.clf()
    #plt.show()


def reprocess_graphs(processed_files_dir, file_seqs, delete_old_files: bool, start_index=0, frames_per_simulation=55):
    print("Re-processing dataset...")
    if delete_old_files:
        for filename in os.listdir(processed_files_dir):
            file_path = os.path.join(processed_files_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    start_file_number = start_index * (frames_per_simulation - 2)
    for i in range(start_file_number, len(file_seqs)):
        geometry_sequence: gl.GeometrySequence = gl.load_geometry_sequence(geoPath, file_seqs[i].base_name,
                                                                            file_seqs[i].index, file_seqs[i].sequence_length)
        if file_seqs[i].index == 1: # new mesh
            connectivity = gl.compute_connectivity(geometry_sequence.mesh)

        g = gl.encode_preprocess(geometry_sequence.input_positions_sequence, geometry_sequence.mesh, geometry_sequence.point_types, connectivity)

        g["point"].targets = geometry_sequence.target_positions
        g["point"].point_types = geometry_sequence.point_types
        g["point"].original_inputs = geometry_sequence.input_positions_sequence

        th.save(g, Path(processed_files_dir, f"{i}.txt"))
        if i % 2000 == 0:
            print(f"{int(i / len(file_seqs) * 100)}%")


def make_multigraph(heterograph, node_norm, mesh_edge_norm, world_edge_norm):
    mesh_edges = gl.EdgeSet(
                    name='mesh_edges',
                    features= mesh_edge_norm(heterograph["point", "mesh", "point"].edge_attr.to(device)),
                    receivers=heterograph["point", "mesh", "point"].edge_index[0].to(device),
                    senders=heterograph["point", "mesh", "point"].edge_index[1].to(device))

    world_edges = gl.EdgeSet(
            name='world_edges',
            features= world_edge_norm(heterograph["point", "world", "point"].edge_attr.to(device)),
            receivers=heterograph["point", "world", "point"].edge_index[0].to(device),
            senders=heterograph["point", "world", "point"].edge_index[1].to(device))
    
    multigraph = gl.MultiGraph(node_features= node_norm(heterograph["point"].x.to(device)), edge_sets=[mesh_edges, world_edges])

    return multigraph


noise_std = 1e-3 * 2
noise_gamma = 0.1
model_path = "models"
checkpoint_path = "models/checkpoints"

if __name__ == "__main__":
    print('Loading', device, th.cuda.current_device(), th.cuda.get_device_name(th.cuda.current_device()))
    geoPath = '../geo'
    plot_path = '../learning_plots/acceleration'
    preprocessed_dataset_path = '../processed_geo/cloth'

    sequence_length = 3
    learning_rate = 1e-2
    min_learning_rate = 1e-6
    batch_size = 1
    num_epochs = 5
    decay_rate = 0.93

    reprocess_dataset = True
    delete_old_files = False

    start_from_checkpoint = False
    start_checkpoint_name = ""

    epochs_to_save_checkpoint = 3

    file_sequences = []
    for root, dirs, files in os.walk(geoPath):
        dirs = sorted(dirs, key=lambda x: int("".join([i for i in x if i.isdigit()])))
        for dir in dirs:
            files = os.listdir(Path(root, dir))
            base_name = files[0].split('.', 1)[0]
            for i in range(1, len(files) - sequence_length + 2):
                seq = HoudiniFileSequence(Path(dir, base_name), i, sequence_length)
                file_sequences.append(seq)

    print(f"We got {len(file_sequences)} sequences to work with")

    if reprocess_dataset:
        reprocess_graphs(preprocessed_dataset_path, file_sequences, delete_old_files, start_index=0, frames_per_simulation=55)

    dataset = SimulationsDataset([Path(preprocessed_dataset_path, f"{index}.txt") for index in range(len(file_sequences))])
    train_test_split = 0.9
    train_dataset = dataset[:int(len(dataset) * train_test_split)]
    test_dataset = dataset[int(len(dataset) * train_test_split):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GraphNetwork().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    lr_lambda = lambda epoch: decay_rate ** epoch if decay_rate ** epoch >= min_learning_rate else min_learning_rate
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda, verbose=True)

    #Normalizer models
    output_normalizer: Normalizer = Normalizer(size=3, name='output_normalizer')
    node_normalizer = Normalizer(size=3 + 3 , name='node_normalizer') # velocity + one hot
    mesh_edge_normalizer = Normalizer(size=8, name='mesh_edge_normalizer')  # 3D rest coord + 3D coord + 2*length = 8
    world_edge_normalizer = Normalizer(size=4, name='world_edge_normalizer')

    normalizers = { 'output': output_normalizer, 'node': node_normalizer, 'mesh_edge': mesh_edge_normalizer, 'world_edge': world_edge_normalizer }

    start_epoch = 0
    if start_from_checkpoint:
        checkpoint = th.load(Path(checkpoint_path, start_checkpoint_name))

        output_normalizer.load_state_dict(checkpoint['output_norm'])
        node_normalizer.load_state_dict(checkpoint['node_norm'])
        mesh_edge_normalizer.load_state_dict(checkpoint['mesh_edge_norm'])
        world_edge_normalizer.load_state_dict(checkpoint['world_edge_norm'])

        model.eval()
        dummy_data = test_loader[0]
        model(make_multigraph(dummy_data, node_normalizer, mesh_edge_normalizer, world_edge_normalizer)) # warming up
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    avg_losses = []
    step = 0
    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        losses = []
        print(f"Epoch {epoch + 1}")
        start_time = time.time()
        p = 0
        for batch_idx, data in enumerate(train_loader):
            mesh_edges = gl.EdgeSet(
                    name='mesh_edges',
                    features= mesh_edge_normalizer(data["point", "mesh", "point"].edge_attr.to(device)),
                    receivers=data["point", "mesh", "point"].edge_index[0].to(device),
                    senders=data["point", "mesh", "point"].edge_index[1].to(device))

            world_edges = gl.EdgeSet(
                    name='world_edges',
                    features= world_edge_normalizer(data["point", "world", "point"].edge_attr.to(device)),
                    receivers=data["point", "world", "point"].edge_index[0].to(device),
                    senders=data["point", "world", "point"].edge_index[1].to(device))
            
            #p += data["point"].x.shape[0]
            #print(data["point"].x.shape[0], data["point", "mesh", "point"].edge_index.shape[1], data["point", "world", "point"].edge_index.shape[1])
            multigraph = gl.MultiGraph(node_features= node_normalizer(data["point"].x), edge_sets=[mesh_edges, world_edges])

            predicted_normalized_acceleration: th.Tensor = model(multigraph)

            accel_targets = output_normalizer(get_target_normalized_acceleration(data["point"].targets, data["point"].original_inputs).to(device))

            error = th.sum((accel_targets - predicted_normalized_acceleration) ** 2, dim=1)
            loss_mask = th.eq(data["point"].point_types.to(device), th.tensor([0], device=device).int())
            loss = th.mean(error[loss_mask])

            if step > 500: # warm up normalizers for first 500 steps
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(float(loss.item()))

            step +=1

            '''if batch_idx % 100 == 0:
                avg_loss = 0.0
                for l in losses:
                    avg_loss += l
                avg_loss = avg_loss / len(losses)
                print(f"Average loss is {avg_loss:.4f}")'''

            #del loss_mask
            #del accel_targets
            #del predicted_normalized_acceleration
            #del error
            #del loss

        avg_loss = th.mean(th.Tensor(losses))
        print(f"Average loss is {avg_loss:.4f}")
        avg_losses.append(avg_loss)
        print(f"Epoch took {(time.time() - start_time):.2f} seconds")

        scheduler.step()
        draw_learning_curve(avg_losses, plot_path, batch_size, len(train_dataset), num_epochs, decay_rate)

        if epoch % epochs_to_save_checkpoint == 0:
            save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, normalizers, avg_loss)

    th.save(model.state_dict(), Path(model_path, "physics_model.pt"))

    node_normalizer.save(Path(model_path, "node_normalizer.pth"))
    mesh_edge_normalizer.save(Path(model_path, "mesh_edge_normalizer.pth"))
    world_edge_normalizer.save(Path(model_path, "world_edge_normalizer.pth"))
    output_normalizer.save(Path(model_path, "output_normalizer.pth"))
    
    #Testing
    model.eval()
    print("Testing")
    test_losses_sum = 0
    num_tests = 0
    for batch_idx, data in enumerate(test_loader):
        mesh_edges = gl.EdgeSet(
                    name='mesh_edges',
                    features= mesh_edge_normalizer(data["point", "mesh", "point"].edge_attr.to(device)),
                    receivers=data["point", "mesh", "point"].edge_index[0].to(device),
                    senders=data["point", "mesh", "point"].edge_index[1].to(device))

        world_edges = gl.EdgeSet(
                name='world_edges',
                features= world_edge_normalizer(data["point", "world", "point"].edge_attr.to(device)),
                receivers=data["point", "world", "point"].edge_index[0].to(device),
                senders=data["point", "world", "point"].edge_index[1].to(device))
        

        multigraph = gl.MultiGraph(node_features= node_normalizer(data["point"].x), edge_sets=[mesh_edges, world_edges])

        predicted_acceleration: th.Tensor = output_normalizer.inverse(model(multigraph))

        predicted_positions = output_postprocessing(data["point"].original_inputs.to(device=device), predicted_acceleration.to(device))

        error = th.sum((data["point"].targets.to(device) - predicted_positions) ** 2, dim=1)
        loss_mask = th.eq(data["point"].point_types.to(device), th.tensor([0], device=device).int())
        loss = th.mean(error[loss_mask])

        test_losses_sum += loss.item()
        num_tests += 1
    print(f"Average test loss is {(test_losses_sum / num_tests):.6f}")

    #input_values = (g.x, g.edge_index, g.edge_attr, geometry_sequence.pinned_points)
    #input_names = ['node_attr', 'edge_index', 'edge_attr', 'particle_types']

    #predicted_normalized_acceleration = model(g.x, g.edge_index, g.edge_attr, geometry_sequence.pinned_points).detach().numpy()
    #print(predicted_normalized_acceleration)

    #torch.onnx.export(model, input_values, "H:\\Animating Tools\\Projects\\Houdini\\LearningPhysics\\scripts\\physics_model.onnx", opset_version=16, input_names=input_names,
                        #output_names=['coords'], dynamic_axes={'node_attr':{0:'num_nodes'}, 'edge_index':{1:'num_edges'}, 'edge_attr':{0:'num_edges'}, 'particle_types':{0:'nu