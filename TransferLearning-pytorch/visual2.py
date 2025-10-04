from graphviz import Digraph

dot = Digraph(comment="Transfer Learning Decision Tree", format="png")

# Root
dot.node('A', "Dataset Size & Task Similarity?")

# Branches
dot.node('B', "<1K images, similar task\n→ Feature Extraction")
dot.node('C', "1K–10K images, similar task\n→ Feature Extraction → Fine-Tuning")
dot.node('D', "10K+ images, different task\n→ Fine-Tuning")
dot.node('E', "100K+ images\n→ Fine-Tune All or Train from Scratch")

# Edges
dot.edge('A', 'B', label="Small dataset")
dot.edge('A', 'C', label="Moderate dataset")
dot.edge('A', 'D', label="Large dataset")
dot.edge('A', 'E', label="Huge dataset")

# Save flowchart
dot.render('TransferLearning-pytorch/transfer_learning_decision_tree', view=True)
