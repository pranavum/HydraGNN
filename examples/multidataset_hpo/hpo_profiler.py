# Profile the Worker Utilization — DeepHyper v0.7.0- documentation
# https://deephyper.readthedocs.io/en/latest/examples/plot_profile_worker_utilization.html

import pandas as pd

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size': 16})

    results = pd.read_csv('results.csv')

    # Remove rows with 'objective' field equal to 'F'
    results = results[results['objective'] != 'F']

    print(results)

    def cumulative_minimum(lst):
        min_value = float('inf')  # Initialize to the smallest possible value
        cumulative_min = []
        for value in lst:
            if value < min_value:
                min_value = value
            cumulative_min.append(min_value)
        return cumulative_min

    def compile_profile(df):
        # Take the results dataframe as input and return the number of jobs running at a given timestamp.
        history = []

        for _, row in df.iterrows():
            history.append((row["m:timestamp_gather"]))

        history = sorted(history)
        nb_workers = 0
        timestamp = [0]
        n_jobs_running = [0]
        for time in history:
            timestamp.append(time)
            n_jobs_running.append(nb_workers)

        return timestamp, n_jobs_running

    plt.figure()

    plt.subplot(1, 1, 1)
    results_objective = [- float(item) for item in results.objective]
    plt.scatter(results["m:timestamp_gather"], results_objective)
    custom_ticks = np.linspace(float(min(results_objective)), float(max(results_objective)), 10, dtype=float)
    plt.plot(results["m:timestamp_gather"],  cumulative_minimum(results_objective))
    plt.xlabel("Time (sec.)")
    plt.ylabel("Validation Loss Function")
    plt.yticks(custom_ticks)
    plt.grid()
    plt.tight_layout()
    plt.show()



