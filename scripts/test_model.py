import torch as th
import geo_loader as gl
import onnxruntime
import os
import train_model as train
from train_model import Normalizer
import hgeo
from pathlib import Path
import time

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

geoPath = 'H:\Animating Tools\Projects\Houdini\LearningPhysics\geo'

sequence_length = 3

test_onnx = False

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

def make_multigraph(inputs, geo_sequence):
    data = gl.encode_preprocess(inputs, geo_sequence.mesh, geo_sequence.point_types, gl.compute_connectivity(geo_sequence.mesh))

    mesh_edges = gl.EdgeSet(
                name='mesh_edges',
                features= mesh_edge_normalizer(data["point", "mesh", "point"].edge_attr),
                receivers=data["point", "mesh", "point"].edge_index[0].to(device),
                senders=data["point", "mesh", "point"].edge_index[1].to(device))

    world_edges = gl.EdgeSet(
            name='world_edges',
            features= world_edge_normalizer(data["point", "world", "point"].edge_attr),
            receivers=data["point", "world", "point"].edge_index[0].to(device),
            senders=data["point", "world", "point"].edge_index[1].to(device))
    

    multigraph = gl.MultiGraph(node_features= node_normalizer(data["point"].x), edge_sets=[mesh_edges, world_edges])

    return multigraph

def load_normalizer(path):
    norm = Normalizer(0, "")
    norm_dict = th.load(path)
    norm._name = norm_dict["name"]
    norm._max_accumulations = norm_dict["max_accumulations"]
    norm._std_epsilon = norm_dict["std_epsilon"]
    norm._acc_count = norm_dict["acc_count"]
    norm._num_accumulations = norm_dict["num_accumulations"]
    norm._acc_sum = norm_dict["acc_sum"]
    norm._acc_sum_squared = norm_dict["acc_sum_squared"]

    return norm


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

    output_normalizer = load_normalizer(Path(model_path, "output_normalizer.pth"))
    node_normalizer = load_normalizer(Path(model_path, "node_normalizer.pth"))
    mesh_edge_normalizer = load_normalizer(Path(model_path, "mesh_edge_normalizer.pth"))
    world_edge_normalizer = load_normalizer(Path(model_path, "world_edge_normalizer.pth"))

    steps = 25

    input_dir = "H:\Animating Tools\Projects\Houdini\LearningPhysics\geo\Sim251"

    start_frame = 1
    geometry_sequence: gl.GeometrySequence = gl.load_geometry_sequence(input_dir, "basic_cloth", start_frame, sequence_length)
    input_sequence = geometry_sequence.input_positions_sequence # nodes, frame, features
    mg = make_multigraph(input_sequence, geometry_sequence)
    #print(mg.edge_sets[1].senders.shape)

    model = train.GraphNetwork().cuda()
    model.eval()
    with th.no_grad():
        model(make_multigraph(input_sequence, geometry_sequence))

        model.load_state_dict(th.load(Path(model_path, "physics_model.pt")))

        input_sequence = input_sequence.cpu()

        for i in range(steps):
            multigraph = make_multigraph(input_sequence, geometry_sequence)
            #print(multigraph.node_features.shape, multigraph.edge_sets[0].features.shape, multigraph.edge_sets[1].features.shape)

            predicted_acceleration: th.Tensor = output_normalizer.inverse(model(multigraph)).cpu()

            predicted_positions = train.output_postprocessing(input_sequence, predicted_acceleration)
            kinematic_mask = th.eq(geometry_sequence.point_types, th.tensor([0]).int())
            
            predicted_positions = th.where(kinematic_mask.view(-1, 1), predicted_positions, input_sequence[:, -1])

            base_mesh : hgeo.Detail = geometry_sequence.mesh
            base_mesh.PointAttributes['P'].Array = predicted_positions.detach().cpu().numpy().tolist()

            base_mesh.save(f"H:\Animating Tools\Projects\Houdini\LearningPhysics\outputs\\251_geo_frame{start_frame + (sequence_length - 1) + i}.geo")

            input_sequence = th.cat([ input_sequence[:, 1:], predicted_positions[:, None, :] ], dim=1)

        print('done')
        #print(model(g.x, g.edge_index, g.edge_attr, geometry_sequence.pinned_points).detach().numpy())
