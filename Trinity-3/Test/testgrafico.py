import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

class AuroraVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.G = nx.DiGraph()
        self.knowledge_nodes = []
        self.inference_paths = []

    def add_knowledge(self, inputs, output, metaM):
        node_id = f"Ms:{output}"
        self.G.add_node(node_id, color='gold', size=500)
        self.knowledge_nodes.append(node_id)
        for i, inp in enumerate(inputs):
            input_id = f"Inp{i}:{inp}"
            self.G.add_node(input_id, color='skyblue', size=300)
            self.G.add_edge(input_id, node_id, label=f"M:{metaM[i]}", color='green')

    def add_inference(self, inputs, output):
        path = []
        for i, inp in enumerate(inputs):
            input_id = f"Inp{i}:{inp}"
            path.append(input_id)
        output_id = f"Inf:{output}"
        self.G.add_node(output_id, color='lightgreen', size=400)
        path.append(output_id)
        self.inference_paths.append(path)

    def visualize(self):
        pos = nx.spring_layout(self.G, seed=42)
        colors = nx.get_node_attributes(self.G, 'color').values()
        sizes = nx.get_node_attributes(self.G, 'size').values()
        nx.draw(self.G, pos, ax=self.ax, node_color=colors, node_size=list(sizes),
                with_labels=True, font_weight='bold')
        edge_labels = nx.get_edge_attributes(self.G, 'label')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        for path in self.inference_paths:
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(self.G, pos, edgelist=path_edges, 
                                  edge_color='red', width=2.0, 
                                  arrowstyle='-|>', arrowsize=20)
        plt.title("Espacio de Conocimiento Fractal de Aurora", fontsize=16)
        plt.show()

    def animate_learning(self, steps):
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(self.G, seed=42)
        def update(frame):
            ax.clear()
            subgraph = self.G.subgraph(self.knowledge_nodes[:frame+1])
            colors = [self.G.nodes[n]['color'] for n in subgraph.nodes]
            sizes = [self.G.nodes[n]['size'] for n in subgraph.nodes]
            nx.draw(subgraph, pos, ax=ax, node_color=colors, node_size=sizes,
                    with_labels=True, font_weight='bold')
            ax.set_title(f"Paso de Aprendizaje {frame+1}/{steps}", fontsize=16)
        return FuncAnimation(fig, update, frames=steps, interval=1500)

# =====================
# DEMOSTRACIÓN PRÁCTICA
# =====================

if __name__ == "__main__":
    vis = AuroraVisualizer()

    # Fase 1: Aurora aprende patrones médicos
    inputs1 = [[1,0,0], [0,1,0]]  # Síntomas: fiebre + tos
    output1 = [1,0,1]              # Diagnóstico: gripe
    metaM1 = [1,1,0]               # Patrón lógico
    vis.add_knowledge(inputs1, output1, metaM1)

    inputs2 = [[0,1,1], [1,0,0]]  # Síntomas: dolor cabeza + fiebre
    output2 = [1,1,0]              # Diagnóstico: migraña
    metaM2 = [0,1,1]
    vis.add_knowledge(inputs2, output2, metaM2)

    # Fase 2: Aurora aprende patrones financieros
    inputs3 = [[1,1,0], [0,0,1]]  # Transacción + historial
    output3 = [1,0,0]              # Riesgo: alto
    metaM3 = [1,0,1]
    vis.add_knowledge(inputs3, output3, metaM3)

    # Fase 3: Aurora realiza inferencias
    vis.add_inference([[1,0,0], [0,1,0]], [1,0,1])  # Infiere gripe
    vis.add_inference([[1,1,0], [0,0,1]], [1,0,0])  # Infiere riesgo

    # Visualización estática
    vis.visualize()
