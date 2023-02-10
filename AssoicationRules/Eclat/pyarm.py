# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 16:27:00 2022

@author: rupak
"""
from PyARMViz import datasets
rules = datasets.load_shopping_rules()


from PyARMViz import datasets
from PyARMViz import PyARMViz

rules = datasets.load_shopping_rules()
PyARMViz.metadata_scatter_plot(rules)

#network diagrams
from PyARMViz import datasets
from PyARMViz import PyARMViz

rules = datasets.load_shopping_rules()
adjacency_graph_plotly(rules)



from PyARMViz import datasets
from PyARMViz import PyARMViz

rules = datasets.load_shopping_rules()
adjacency_graph_gephi(rules)