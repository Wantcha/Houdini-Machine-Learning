from typing import NamedTuple, OrderedDict
from unicodedata import normalize
import hgeo
import hjson
import numpy as np
import torch as th
import torch_geometric.data as tgd
import collections
import os

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

def encodePreprocess(input_position_sequence : th.Tensor, mesh: hgeo.Detail) -> tgd.Data:
    point_positions = input_position_sequence[-1]

    velocity_sequence = input_position_sequence[1:] - input_position_sequence[:-1]
    normalized_velocity_sequence = (velocity_sequence - VELOCITY_STATS.mean) / VELOCITY_STATS.std

    normalized_velocity_sequence = normalized_velocity_sequence.permute(1, 0, 2) # shape should now be (npoints, seq_length, dimensions)
    normalized_velocity_sequence = normalized_velocity_sequence.contiguous().view(normalized_velocity_sequence.shape[0], -1) # shape should now be (npoints, features)
    # print('size of normalized velocity sequence:')
    # print(normalized_velocity_sequence.size())

    sourceNodes = []
    destNodes = []
    edgeLengths = []
    edgeDisplacements = []
    
    for p in mesh.Primitives:
        for i in range(1, len(p.Vertices)):
            p1 = mesh.vertexPoint(p.Vertices[i-1])
            p2 = mesh.vertexPoint(p.Vertices[i])
            sourceNodes.extend( (p1, p2) )
            destNodes.extend( (p2, p1) )
            #sourceNodes.append(p1)
            #destNodes.append(p2) #
            #sourceNodes.append(p2)
            #destNodes.append(p1)
            displacement = point_positions[p2] - point_positions[p1]
            edgeDisplacements.extend( (displacement, displacement) )
            l = th.tensor(np.linalg.norm( displacement ), dtype=th.float32) # edge length
            edgeLengths.extend( (l, l) )

        p1 = mesh.vertexPoint(p.Vertices[len(p.Vertices) - 1])
        p2 = mesh.vertexPoint(p.Vertices[0])
        sourceNodes.extend( (p1, p2) )
        destNodes.extend( (p2, p1) )
        #sourceNodes.append(p1)
        #destNodes.append(p2) #
        #sourceNodes.append(p2)
        #destNodes.append(p1)
        displacement = point_positions[p2] - point_positions[p1]
        edgeDisplacements.extend( (displacement, displacement) )
        l = th.tensor(np.linalg.norm( displacement ), dtype=th.float32)
        edgeLengths.extend( (l, l) )

    edge_index = th.tensor([sourceNodes, destNodes], dtype=th.long)

    
    #g.ndata['Feats'] = normalized_velocity_sequence
    #g.ndata['Type'] = th.tensor(pinned_points, dtype=th.float32) # 1-Free; 2-Pinned
    #g.edata['Feats'] = th.tensor(edgeLengths, dtype=th.float32)
    edgeDisplacements = th.stack(edgeDisplacements)
    edgeLengths = th.stack(edgeLengths)
    
    edge_attr = th.cat((edgeDisplacements, edgeLengths.unsqueeze(-1)), 1)

    if (len(edge_attr.shape) == 1):
            adjustedfeats = []
            for i in range(edge_attr.shape[0]):
                adjustedfeats.append( edge_attr[i].unsqueeze(dim=0) )
            edge_attr = th.stack(adjustedfeats)

    g = tgd.Data(x=normalized_velocity_sequence, edge_index=edge_index, edge_attr=edge_attr)
    return g


def calculate_noisy_velocities(positions: th.Tensor, sampled_noise):
    noisy_positions = positions + sampled_noise
    
    velocity_sequence = noisy_positions[1:] - noisy_positions[:-1]
    normalized_velocity_sequence = (velocity_sequence - VELOCITY_STATS.mean) / VELOCITY_STATS.std

    normalized_velocity_sequence = normalized_velocity_sequence.permute(1, 0, 2) # shape should now be (npoints, seq_length, dimensions)
    normalized_velocity_sequence = normalized_velocity_sequence.contiguous().view(normalized_velocity_sequence.shape[0], -1) # shape should now be (npoints, features)

    return normalized_velocity_sequence


def get_random_walk_noise_for_position_sequence(position_sequence : th.Tensor, noise_std_last_step):
    """Returns random-walk noise in the velocity applied to the position."""

    velocity_sequence = position_sequence[1:] - position_sequence[:-1]

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
    position_sequence_noise = th.concat([ th.zeros_like(velocity_sequence_noise[0:1,:,:]),
                                           th.cumsum(velocity_sequence_noise, dim=1)], dim=0)

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
        mesh.loadJSON(hjson.load(f))
    return mesh

def load_geometry_sequence(filepath: str, base_name: str, start_frame: int, sequence_length: int) -> GeometrySequence:
    position_sequence = []

    for b in range(start_frame, start_frame + sequence_length - 1):
        mesh = load_detail(f"{filepath}\\{base_name}.{b}.geo")
        # npoints = mesh.pointCount
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

    input_position_sequence = positions_tensor[:-1]
    target_position = positions_tensor[-1]

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

