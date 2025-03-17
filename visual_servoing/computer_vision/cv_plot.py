import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import random

def read_scores_by_algorithm(filename):
    """
    Reads the CSV file and returns a dictionary mapping each algorithm
    to a list of scores.
    Assumes the CSV file has headers: "Algorithm", "Image", "Score".
    """
    data = defaultdict(list)
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            algo = row['Algorithm']
            score = float(row['Score'])
            data[algo].append(score)
    return data

def main():
    # Determine the path to the CSV file (adjust if needed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "scores", "test_scores_citgo.csv")
    
    # Read the data grouped by algorithm
    data = read_scores_by_algorithm(csv_path)
    
    # Determine the order for the algorithms (e.g., sorted alphabetically)
    algorithms = sorted(data.keys())
    
    # Create a figure
    plt.figure(figsize=(8, 6))
    
    # Set x positions (1, 2, 3, ...) for each algorithm category
    x_positions = list(range(1, len(algorithms) + 1))
    
    for i, algo in enumerate(algorithms):
        x_pos = i + 1  # x-coordinate for this algorithm
        scores = data[algo]
        
        # Create jitter for x-values so that dots don't overlap
        jitter = 0.08  # maximum jitter in either direction
        x_vals = [x_pos + random.uniform(-jitter, jitter) for _ in scores]
        
        # Plot each score as a blue dot
        plt.scatter(x_vals, scores, color='blue', alpha=0.7)
        
        # Compute the average score for this algorithm
        avg_score = sum(scores) / len(scores)
        # Plot a short horizontal line (0.2 units wide) at the average score in green
        plt.plot([x_pos - 0.1, x_pos + 0.1], [avg_score, avg_score],
                 color='green', linewidth=3)
    
    # Set x-axis tick positions and labels
    plt.xticks(x_positions, algorithms)
    plt.xlabel("Algorithm")
    plt.ylabel("Score")
    plt.title("Citgo Scores by Algorithm with Average Indicator")
    plt.grid(True)
    
    # Save the plot as a file
    output_path = os.path.join(script_dir, "citgo_scores.png")
    plt.savefig(output_path)
    print(f"Plot saved as {output_path}")
    plt.close()

if __name__ == '__main__':
    main()
