import torch
import gym
import pandas as pd


def Q1_algo(vec1,vec2):
    """function does stuff"""
    n = len(vec1)
    m = len(vec2)
    all_values = set(vec1).union(set(vec2))
    val_df = pd.DataFrame(columns=all_values)
    indices = pd.Series([0]*len(all_values),index=all_values)
    counter = n
    for val in vec1:
        val_df.loc[indices[val], val] = counter
        indices[val] += 1
        counter -= 1

    # filling zeros
    val_df = val_df.fillna(0)
    for key in range(len(val_df),m):
        val_df.loc[key] = 0

    options_df = pd.DataFrame()
    for e in enumerate(vec2):
        options_df.loc[:,str(e[0])] = (val_df.loc[:, e[1]])

    for c in options_df.columns:
        if all(options_df[c] == 0):
            options_df = options_df.drop(c,axis=1)

    for r in options_df.index:
        if all(options_df.loc[r] == 0):
            options_df = options_df.drop(r,axis=0)

    options_df

    return options_df

# ans is VBACD
vec1 = ['A','V','B','V','A','M','C','D']
vec2 = ['W','V','Z','B','Q','A','C','A','D']
Q1_algo(vec1, vec2)

