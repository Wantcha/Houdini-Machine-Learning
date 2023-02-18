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

sequence_length = 3
particle_type_embedding_size = train.particle_type_embedding_size

test_onnx = False

device = train.device

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

    model = train.GraphNetwork(particle_type_embedding_size).to(device)
    model.load_state_dict(th.load(model_path))

    # warming up parameters
    '''num_edges = np.random.randint(1000, 5000)
    num_nodes = 300
    randedge_index = th.randint(0, num_nodes, (2, num_edges))
    randedge_attr = th.rand(num_edges, 1)
    randx = th.rand(num_nodes, 9)
    pinned_points = th.randint(0, 1, (num_nodes,))

    model(randx, randedge_index, randedge_attr, pinned_points)'''
    # warmed up

    #model.load_state_dict(th.load(model_path))
    steps = 20

    model.eval()

    input_dir = "H:\Animating Tools\Projects\Houdini\LearningPhysics\geo\Sim251"

    start_frame = 1
    geometry_sequence: gl.GeometrySequence = gl.load_geometry_sequence(input_dir, "basic_cloth", start_frame, sequence_length)
    input_sequence = geometry_sequence.input_positions_sequence # nodes, frame, features

    for i in range(steps):
        g = gl.encodePreprocess(input_sequence, geometry_sequence.mesh)

        predicted_normalized_acceleration = model(g.x.to(device), g.edge_index.to(device), g.edge_attr.to(device), geometry_sequence.pinned_points.to(device))

        predicted_position = train.output_postprocessing(geometry_sequence.input_positions_sequence.to(device), predicted_normalized_acceleration)

        kinematic_mask = geometry_sequence.pinned_points
        kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, 3)

        predicted_position = th.where(kinematic_mask, predicted_position, geometry_sequence.input_positions_sequence[:, 0])

        base_mesh : hgeo.Detail = geometry_sequence.mesh
        #print(len(base_mesh.PointAttributes['P'].Array))
        base_mesh.PointAttributes['P'].Array = predicted_position.detach().numpy().tolist()
        #base_mesh.Primitives[0]
        #print(len(base_mesh.PointAttributes['P'].Array))

        base_mesh.save(f"H:\Animating Tools\Projects\Houdini\LearningPhysics\outputs\\251_geo_frame{start_frame + (sequence_length - 1) + i}.geo")

        input_sequence = th.cat([ input_sequence[:, 1:], predicted_position ], dim=1)


    #print(model(g.x, g.edge_index, g.edge_attr, geometry_sequence.pinned_points).detach().numpy())
