import os
import hydragnn

filepath = os.path.join(os.path.dirname(__file__), "eam.json")
hydragnn.run_training(filepath)
