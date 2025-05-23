import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
import torch
import torch_geometric.utils
import tqdm

from torch_geometric.loader import DataLoader

from route_choice.data.tutorial import draw_networkx_edge_attr
from route_choice.recursive_logit import RecursiveLogit


@pytest.mark.parametrize(
    "rl_tutorial_dataset,use_vi,plot",
    [({"n_samples": 1000, "seed": 321}, True, False), ({"n_samples": 1000, "seed": 321}, False, True)],
    indirect=["rl_tutorial_dataset"],
)
def test_rl_tutorial_dataset(rl_tutorial_dataset, use_vi, plot):
    dataset, feat_scaler, n_feats = rl_tutorial_dataset
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    model = RecursiveLogit(n_feats, link_constant=True, use_value_iteration=use_vi)
    optim = torch.optim.Adam(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, threshold=1e-4, threshold_mode="rel", patience=50, min_lr=1e-4
    )

    progress_bar = tqdm.trange(200)
    for epoch in progress_bar:
        epoch_loss = 0
        for batch in loader:
            feats_scaled_np = feat_scaler.transform(batch.edge_attr.numpy())
            batch.feats = torch.as_tensor(feats_scaled_np, dtype=torch.float32)
            batch.choice = batch.path

            model.train()
            optim.zero_grad()

            loss, info = model.train_step(batch, loss_reduction="sum")
            loss.backward()
            optim.step()

            epoch_loss += loss.detach()

        scheduler.step(epoch_loss)
        progress_bar.set_postfix({"loss": epoch_loss.item(), "lr": scheduler.get_last_lr()[0]})

    params = model.get_params()
    beta = params["coeffs.weight"].numpy() / feat_scaler.scale_
    lc = params["coeffs.bias"].numpy() - params["coeffs.weight"].numpy() * feat_scaler.mean_ / feat_scaler.scale_
    assert np.isclose(beta, np.array([-2.0]), rtol=0.1), f"beta did not match expected value of -2 ({beta})"
    assert np.isclose(lc, np.array([-0.01]), atol=0.05), f"link constant did not match expected value of -0.01 ({lc})"
    assert lc < 0, f"expected link constant to be negative but got {lc}"

    probs = torch_geometric.utils.to_dense_adj(batch.edge_index, batch.batch, info["prob"].squeeze(-1))
    in_flows, _ = torch_geometric.utils.to_dense_batch(batch.is_orig, batch.batch)

    batch_size, n_nodes = in_flows.shape
    I = torch.eye(n_nodes).expand(batch_size, -1, -1)
    flows = torch.linalg.solve(I - probs.permute(0, 2, 1), in_flows * 100)

    state_list = list(dataset.state_graph.nodes)
    for u, v, k, flow in dataset.source_graph.edges(keys=True, data="flow"):
        state_idx = state_list.index((u, v, k))
        assert torch.isclose(
            flows[:, state_idx], torch.as_tensor(flow), rtol=0.1
        ).all(), f"flow on link {u, v, k} did not match the expected value of {flow}"

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        dataset.plot_dataset(axes[0], axes[1])

        edge_flow = {}
        for i, k in enumerate(dataset.state_graph.nodes):
            edge_flow[k] = flows[:, i].mean()
        edge_flow.pop(dataset.source_graph.graph["orig"])
        edge_flow.pop(dataset.source_graph.graph["dest"])

        node_pos = nx.get_node_attributes(dataset.source_graph, "pos")
        node_labels = {n: n for n in dataset.source_graph.nodes}

        axes[2].set_title("Flow (Estimated)")
        nx.draw(dataset.source_graph, node_pos, labels=node_labels, ax=axes[2], edgelist=[])
        draw_networkx_edge_attr(dataset.source_graph, node_pos, edge_flow, ax=axes[2], cmap="viridis")

        fig.savefig("/tmp/test_rl_tutorial_dataset.png")
