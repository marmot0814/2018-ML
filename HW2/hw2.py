import numpy as np
import pandas as pd
from process import process
from matplotlib import pyplot as plt

# data processing
df = pd.read_csv("./datasets/data_noah.csv")
df = df[['x', 'y', 'pitch_type']]
