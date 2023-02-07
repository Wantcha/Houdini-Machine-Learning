import numpy as np
import torch as th
import torch.nn as nn
import geo_loader as gl
import onnxruntime
import os
import train_model as train
import hgeo

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

geoPath = 'H:\Animating Tools\Projects\Houdini\LearningPhysics\geo'

sequence_length = 5
particle_type_embedding_size = 8

geometry_sequence = gl.load_geometry_sequences(geoPath, sequence_length)    
g = gl.encodePreprocess(geometry_sequence.input_positions_sequence, geometry_sequence.mesh)

test_onnx = False

if test_onnx:
    ort_session = onnxruntime.InferenceSession('physics_model.onnx')

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = { ort_session.get_inputs()[0].name: to_numpy(g.x), ort_session.get_inputs()[1].name: to_numpy(g.edge_index),
                    ort_session.get_inputs()[2].name: to_numpy(g.edge_attr), ort_session.get_inputs()[3].name: to_numpy(geometry_sequence.pinned_points) }
    ort_outs = ort_session.run(None, ort_inputs)

    output = ort_outs[0]

    print(output)

else:
    model_path = train.model_path

    model = train.GraphNetwork(particle_type_embedding_size)

    # warming up parameters
    num_edges = np.random.randint(1000, 5000)
    num_nodes = 300
    randedge_index = th.randint(0, num_nodes, (2, num_edges))
    randedge_attr = th.rand(num_edges, 1)
    randx = th.rand(num_nodes, 9)
    pinned_points = th.randint(0, 1, (num_nodes,))

    model(randx, randedge_index, randedge_attr, pinned_points)
    # warmed up

    model.load_state_dict(th.load(model_path), strict=False)
    model.eval()

    predicted_normalized_acceleration = model(g.x, g.edge_index, g.edge_attr, geometry_sequence.pinned_points).detach().numpy()
    predicted_acceleration = (predicted_normalized_acceleration * train.ACCELERATION_STATS.std) + train.ACCELERATION_STATS.mean

    most_recent_position : th.Tensor = geometry_sequence.input_positions_sequence[-1]
    most_recent_velocity = most_recent_position - geometry_sequence.input_positions_sequence[-2]

    new_velocity = most_recent_velocity + predicted_acceleration
    new_position : th.Tensor = most_recent_position + new_velocity

    base_mesh : hgeo.Detail = geometry_sequence.mesh
    #print(len(base_mesh.PointAttributes['P'].Array))
    base_mesh.PointAttributes['P'].Array = new_position.detach().numpy().tolist()
    base_mesh.Primitives[0]
    #print(len(base_mesh.PointAttributes['P'].Array))

    base_mesh.save("H:\Animating Tools\Projects\Houdini\LearningPhysics\outputs\\test_geo.geo")


    #print(model(g.x, g.edge_index, g.edge_attr, geometry_sequence.pinned_points).detach().numpy())
