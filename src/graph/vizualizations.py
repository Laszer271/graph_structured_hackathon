from streamlit_agraph import agraph, Node, Edge, Config

from .schema import BaseSchema, FolderSchema, DocumentSchema, ChunkSchema, EntitySchema

class GraphData:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node: BaseSchema):
        node = Node(
            id=len(self.nodes),
            size=25,
            shape="circularImage",
            **node.to_dict(),
        )
        self.nodes.append(node)

    def add_edge(self, label, source_id, target_id):
        edge = Edge(
            source=source_id,
            label=label,
            target=target_id,
        )
        self.edges.append(edge)


config = Config(width=750,
                height=950,
                directed=True, 
                physics=True, 
                hierarchical=False,
                # **kwargs
                )

return_value = agraph(nodes=nodes, 
                      edges=edges, 
                      config=config)