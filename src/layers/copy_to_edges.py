from torch_geometric.nn import MessagePassing


class CopyToEdges(MessagePassing):
    def __init__(self, dir: str = "to_incoming"):
        # "to_incoming" copies values backwards, i.e. onto incoming edges
        # "to_outgoing" does the opposite
        if dir == "to_incoming":
            flow = "target_to_source"
        elif dir == "to_outgoing":
            flow = "source_to_target"
        else:
            raise ValueError(f"invalid dir: {dir}")
        super().__init__(aggr=None, flow=flow)

    def forward(self, x, edge_index):
        # it's poorly documented, but there is a function self.edge_updater does something similar to self.propagate for edges
        # where self.edge_update corresponds to self.message
        return self.edge_updater(edge_index, x=x)

    def edge_update(self, x_j):
        return x_j
