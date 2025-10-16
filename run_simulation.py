from analyze import *
import pickle

all_results = run_all()

with open("simulation_results.pkl", 'wb') as file:
    pickle.dump(all_results)