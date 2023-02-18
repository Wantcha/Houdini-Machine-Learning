from typing import NamedTuple, OrderedDict
from unicodedata import normalize
import hgeo
import hjson
import numpy as np
import torch as th
import torch_geometric.data as tgd
import collections
import os
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

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


def encodePreprocess(input_position_sequence : th.Tensor, mesh: hgeo.Detail, point_types: th.Tensor, vertex_radius = 0.5) -> tgd.Data:
    point_positions = input_position_sequence[:, -1]

    velocity_sequence = time_diff(input_position_sequence).to(device)
    #normalized_velocity_sequence = (velocity_sequence - VELOCITY_STATS.mean) / VELOCITY_STATS.std

    velocity_sequence = velocity_sequence.contiguous().view(velocity_sequence.shape[0], -1) # shape should now be (npoints, features)
    total_point_types = 2
    node_types = th.nn.functional.one_hot(point_types.long(), total_point_types).to(device)
    node_features = th.cat([velocity_sequence, node_types], dim=-1)
    mesh_edges = {}

    mesh_sourceNodes = []
    mesh_destNodes = []
    
    for p in mesh.Primitives:
        for i in range(1, len(p.Vertices)):
            p1 = mesh.vertexPoint(p.Vertices[i-1])
            p2 = mesh.vertexPoint(p.Vertices[i])

            if (p1, p2) not in mesh_edges:
                mesh_edges[(p1, p2)] = True
                mesh_edges[(p2, p1)] = True

                mesh_sourceNodes.extend( (p1, p2) )
                mesh_destNodes.extend( (p2, p1) )

        p1 = mesh.vertexPoint(p.Vertices[len(p.Vertices) - 1])
        p2 = mesh.vertexPoint(p.Vertices[0])

        if (p1, p2) not in mesh_edges:
            mesh_edges[(p1, p2)] = True
            mesh_edges[(p2, p1)] = True

            mesh_sourceNodes.extend( (p1, p2) )
            mesh_destNodes.extend( (p2, p1) )

    #edge_index = th.tensor([mesh_sourceNodes, mesh_destNodes], dtype=th.long)
    pattributes_dict = mesh.PointAttributes
    rest_positions = th.tensor(pattributes_dict['rest'].Array)
    #print(rest_positions)
    
    relative_world_displacements = point_positions[mesh_sourceNodes, :] - point_positions[mesh_destNodes, :]
    relative_world_distances = th.norm(relative_world_displacements, dim=-1, keepdim=True)

    relative_rest_displacements = rest_positions[mesh_sourceNodes, :] - rest_positions[mesh_destNodes, :]
    relative_rest_distances = th.norm(relative_world_displacements, dim=-1, keepdim=True)

    mesh_edge_features = [relative_rest_displacements, relative_rest_distances, relative_world_displacements, relative_world_distances]
    mesh_edge_features = th.cat(mesh_edge_features, dim = -1)

    world_sourceNodes = []
    world_destNodes = []
    world_edge_features = []
    world_edges = {}
    
    proximity_points = pattributes_dict['close_points'].Array
    for i in range(len(proximity_points)):
        close_pts = proximity_points[i]
        if len(close_pts):
            for point in close_pts:
                if (i, point) not in world_edges:
                    world_edges[(i, point)] = True
                    world_edges[(point, i)] = True

                    world_sourceNodes.extend ( (i, point) )
                    world_destNodes.extend ( (point, i) )

    world_displacements = point_positions[world_sourceNodes, :] - point_positions[world_destNodes, :]
    world_edge_features.append(world_displacements)

    world_distances = th.norm(world_displacements, dim=-1, keepdim=True)
    world_edge_features.append(world_distances)

    world_edge_features = th.cat(world_edge_features, dim = -1)


    #g = tgd.Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    g = tgd.HeteroData()
    g["point"].x = node_features
    g["point", "mesh", "point"].edge_index = th.tensor([mesh_sourceNodes, mesh_destNodes], dtype=th.long)
    g["point", "mesh", "point"].edge_attr = mesh_edge_features
    g["point", "world", "point"].edge_index = th.tensor([world_sourceNodes, world_destNodes], dtype=th.long)
    g["point", "world", "point"].edge_attr = world_edge_features
    return g


def time_diff(input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]


def calculate_noisy_velocities(positions: th.Tensor, sampled_noise):
    noisy_positions = positions + sampled_noise
    
    velocity_sequence = time_diff(noisy_positions)
    #normalized_velocity_sequence = (velocity_sequence.to('cuda') - VELOCITY_STATS.mean) / VELOCITY_STATS.std

    normalized_velocity_sequence = velocity_sequence.contiguous().view(velocity_sequence.shape[0], -1) # shape should now be (npoints, features)

    return normalized_velocity_sequence


def get_random_walk_noise_for_position_sequence(position_sequence : th.Tensor, noise_std_last_step):
    """Returns random-walk noise in the velocity applied to the position."""

    velocity_sequence = time_diff(position_sequence)

    # We want the noise scale in the velocity at the last step to be fixed.
    # Because we are going to compose noise at each step using a random_walk:
    # std_last_step**2 = num_velocities * std_each_step**2
    # so to keep `std_last_step` fixed, we apply at each step:
    # std_each_step `std_last_step / np.sqrt(num_input_velocities)`
    num_velocities = velocity_sequence.shape[1]
    noise_sampler = th.distributions.Normal(loc=0, scale=noise_std_last_step / num_velocities ** 0.5)
    velocity_sequence_noise = noise_sampler.sample(velocity_sequence.shape)

    # Apply the random walk.
    velocity_sequence_noise = th.cumsum(velocity_sequence_noise, dim=1) #tf.cumsum(velocity_sequence_noise, axis=1)
    #print(th.zeros_like(velocity_sequence_noise[0:1,:,:]).shape)
    #print(th.cumsum(velocity_sequence_noise, dim=1).shape)

    # Integrate the noise in the velocity to the positions, assuming
    # an Euler intergrator and a dt = 1, and adding no noise to the very first
    # position (since that will only be used to calculate the first position
    # change).
    position_sequence_noise = th.concat([ th.zeros_like(velocity_sequence_noise[:, 0:1]),
                                           th.cumsum(velocity_sequence_noise, dim=1)], dim=1)
    
    return position_sequence_noise

class GeometrySequence(NamedTuple):
    input_positions_sequence: th.Tensor
    target_positions: th.Tensor
    num_points: int
    pinned_points: th.Tensor
    mesh: hgeo.Detail

def load_detail(filepath: str) -> hgeo.Detail:
    with open(filepath, 'r') as f:
        mesh = hgeo.Detail()
        #print(filepath)
        mesh.loadJSON(hjson.load(f))
    return mesh

def load_geometry_sequence(filepath: str, base_name: str, start_frame: int, sequence_length: int) -> GeometrySequence:
    position_sequence = []

    for b in range(start_frame, start_frame + sequence_length - 1):
        mesh = load_detail(f"{filepath}\\{base_name}.{b}.geo")
        pattributes_dict = mesh.PointAttributes
        position_sequence.append(th.tensor(pattributes_dict['P'].Array, dtype=th.float32))
    
    mesh = load_detail(f"{filepath}\\{base_name}.{start_frame + sequence_length - 1}.geo")

    # Get point types
    pattributes_dict = mesh.PointAttributes
    position_sequence.append(th.tensor(pattributes_dict['P'].Array))

    npoints = len(pattributes_dict['P'].Array)

    pgroups_dict = mesh.PointGroups
    if 'pinned' in pgroups_dict:
        pinned_points = th.tensor(np.array(pgroups_dict['pinned'].Selection), dtype=th.int32)
    else:
        pinned_points = th.zeros(npoints, dtype=th.int32)

    positions_tensor = th.stack(position_sequence)
    positions_tensor = positions_tensor.permute(1, 0, 2) # shape should now be (npoints, seq_length, dimensions)

    input_position_sequence = positions_tensor[:, :-1]
    target_position = positions_tensor[:, -1]

    return GeometrySequence(input_position_sequence, target_position, npoints, pinned_points, mesh)


if __name__ == "__main__":
    '''print('Loading')
    geoPath = 'H:\Animating Tools\Projects\Houdini\LearningPhysics\geo'

    # print(input_position_sequence.size())
    
    g = encodePreprocess(input_position_sequence)

    model = GraphNetwork(th.tensor(pinned_points, dtype=th.int32), particle_type_embedding_size)
    # print(g.x.shape, g.edge_attr.shape, g.edge_index.shape, pinned_points.shape)
    input_values = (g.x, g.edge_index, g.edge_attr)
    input_names = ['node_attr', 'edge_index', 'edge_attr']

    predicted_normalized_acceleration = model(g.x, g.edge_index, g.edge_attr).detach().numpy()
    
    #print(new_g.edata['EncodedFeats'].shape)
    #print(predicted_accs.shape)
    predicted_acceleration = (predicted_normalized_acceleration * ACCELERATION_STATS.std) + ACCELERATION_STATS.mean

    most_recent_position = input_position_sequence[-1]
    most_recent_velocity = most_recent_position - input_position_sequence[-2]

    new_velocity = most_recent_velocity + predicted_acceleration
    new_position = most_recent_position + new_velocity

    position_sequence_noise = get_random_walk_noise_for_position_sequence(input_position_sequence, noise_std_last_step=6.7e-4)'''

