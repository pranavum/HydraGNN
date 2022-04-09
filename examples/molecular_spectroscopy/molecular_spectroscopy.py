import os
import hydragnn

filepath = os.path.join(os.path.dirname(__file__), "molecular_spectroscopy.json")
hydragnn.run_training(filepath)
