# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 16:10:52 2022

@author: rupak
"""

#pip install pyECLAT

# store the item sets as lists of strings in a list
transactions = [
    ['beer', 'wine', 'cheese'],
    ['beer', 'potato chips'],
    ['eggs', 'flower', 'butter', 'cheese'],
    ['eggs', 'flower', 'butter', 'beer', 'potato chips'],
    ['wine', 'cheese'],
    ['potato chips'],
    ['eggs', 'flower', 'butter', 'wine', 'cheese'],
    ['eggs', 'flower', 'butter', 'beer', 'potato chips'],
    ['wine', 'beer'],
    ['beer', 'potato chips'],
    ['butter', 'eggs'],
    ['beer', 'potato chips'],
    ['flower', 'eggs'],
    ['beer', 'potato chips'],
    ['eggs', 'flower', 'butter', 'wine', 'cheese'],
    ['beer', 'wine', 'potato chips', 'cheese'],
    ['wine', 'cheese'],
    ['beer', 'potato chips'],
    ['wine', 'cheese'],
    ['beer', 'potato chips']
]

import pandas as pd

# you simply convert the transaction list into a dataframe
data = pd.DataFrame(transactions)
data

# we are looking for itemSETS
# we do not want to have any individual products returned
min_n_products = 2

# we want to set min support to 7
# but we have to express it as a percentage
min_support = 7/len(transactions)

# we have no limit on the size of association rules
# so we set it to the longest transaction
max_length = max([len(x) for x in transactions])

from pyECLAT import ECLAT

# create an instance of eclat
eclat_rules = ECLAT(data=data, verbose=True)

# fit the algorithm
rule_indices, rule_supports = eclat_rules.fit(min_support=min_support,
                                           min_combination=min_n_products,
                                           max_combination=max_length)


print(rule_supports)