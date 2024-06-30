from typing import List, Dict
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from streamlit_agraph import agraph, Node, Edge, Config
from neo4j import GraphDatabase

from graph.colors import colors

from dotenv import load_dotenv
load_dotenv()


class GraphData:
    def __init__(self):
        self.nodes = []
        self.edges = []

        self.nodes_ids = set()
        self.relationships_ids = set()
        self.types_to_colors = dict()

    def add_node(self, node: Dict):
        if node['elem_id'] in self.nodes_ids:
            return
        
        if node['node_type'] not in self.types_to_colors:
            self.types_to_colors[node['node_type']] = colors[len(self.types_to_colors) % len(colors)]
        color = self.types_to_colors[node['node_type']]

        if 'id' in node:
            del node['id']

        graph_node = Node(
            id=node['elem_id'],
            title=node.get('text', None), # displayed if hovered
            label=node.get('name', None), # displayed inside the node
            size=25,
            # shape="circularImage",
            color=color,
            **node,
        )
        self.nodes.append(graph_node)
        self.nodes_ids.add(node['elem_id'])

    def add_edge(self, source_id: str, target_id: str, relationship: Dict):
        if relationship['elem_id'] in self.relationships_ids:
            return

        graph_edge = Edge(
            source=source_id,
            label=relationship['edge_type'],
            target=target_id,
        )
        self.edges.append(graph_edge)
        self.relationships_ids.add(relationship['elem_id'])

    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        return self.edges
    

def _get_all_nodes(tx):
    result = tx.run("MATCH (n)-[r]->(m:Entity {type: 'person'}) RETURN n,r,m")
    return [
        {
            "node1": {
                "elem_id": rec['n'].element_id,
                "type": list(rec['n'].labels)[0],
                **dict(rec['n'])
                },
            "node2": {
                "elem_id": rec['m'].element_id,
                "type": list(rec['m'].labels)[0],
                **dict(rec['m'])
                },
            "relationship": {
                "elem_id": rec['r'].element_id,
                "type": rec['r'].type,
                **dict(rec['r'])
                }
        }
        # rec
        for rec in result
      ]

def make_graph(nodes, relationships=None):
    config = Config(width=1500,
                    height=1000,
                    directed=True, 
                    physics=True, 
                    hierarchical=False,
                    # **kwargs
                    )

    graph = GraphData()
    for node in nodes:
        graph.add_node(node=node)

    for rel in relationships:
        if rel['node1'] is None or rel['node2'] is None:
            continue
        source_id = rel['node1']['elem_id']
        target_id = rel['node2']['elem_id']
        graph.add_edge(source_id=source_id, target_id=target_id, relationship=rel['relationship'])

    return dict(
        nodes=graph.get_nodes(), 
        edges=graph.get_edges(), 
        config=config
    )

if __name__ == '__main__':
    # Neo4j AuraDB connection details
    uri = os.getenv("NEO4J_URI")
    password = os.getenv("NEO4J_KEY")
    username = 'neo4j'

    # Connect to Neo4j AuraDB
    driver = GraphDatabase.driver(uri, auth=(username, password))

    with driver.session() as session:
        relationships_list = session.execute_read(_get_all_nodes)
    driver.close()


    config = Config(width=1500,
                    height=1000,
                    directed=True, 
                    physics=True, 
                    hierarchical=False,
                    # **kwargs
                    )

    graph = GraphData()
    for rel in relationships_list:
        source_id = rel['node1']['elem_id']
        target_id = rel['node2']['elem_id']
        graph.add_node(node=rel['node1'])
        graph.add_node(node=rel['node2'])
        graph.add_edge(source_id=source_id, target_id=target_id, relationship=rel['relationship'])


    return_value = agraph(
        nodes=graph.get_nodes(), 
        edges=graph.get_edges(), 
        config=config
    )