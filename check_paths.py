import os
import pickle
import numpy as np

print("Checking current working directory:", os.getcwd())
print("data/ exists:", os.path.exists("data"))
print("artifacts/ exists:", os.path.exists("artifacts"))

# List contents
if os.path.exists("data"):
    print("data/ contents:", os.listdir("data"))
if os.path.exists("artifacts"): 
    print("artifacts/ contents:", os.listdir("artifacts"))