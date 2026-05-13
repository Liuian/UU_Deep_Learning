import json
import matplotlib.pyplot as plt
import argparse
import os
from collections import defaultdict

def plot_performance(json_path):
    # Load results
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    if not results:
        print("No results found in the JSON file.")
        return

    # Group by sequence length and find the best (minimum) scaled MAE
    seq_length_to_maes = defaultdict(list)
    for res in results:
        seq_length = res.get('seq_length')
        mae_scaled = res.get('mae_scaled')
        if seq_length is not None and mae_scaled is not None:
            seq_length_to_maes[seq_length].append(mae_scaled)

    # Calculate best MAE for each seq_length
    sorted_seq_lengths = sorted(seq_length_to_maes.keys())
    best_maes = [min(seq_length_to_maes[sl]) for sl in sorted_seq_lengths]

    # Style to match other graphs in the project
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150) # Wide aspect ratio like others
    
    # Colors matching the project style (SteelBlue/Tab:Blue)
    line_color = 'steelblue'
    
    # Plot line and points
    ax.plot(sorted_seq_lengths, best_maes, marker='o', linestyle='-', linewidth=1.5, 
            markersize=6, color=line_color, label='Minimum Scaled MAE')
    
    # Annotate points
    for i, (sl, mae) in enumerate(zip(sorted_seq_lengths, best_maes)):
        ax.annotate(f'{mae:.5f}', 
                    (sl, mae), 
                    textcoords="offset points", 
                    xytext=(0, 8), 
                    ha='center', 
                    fontsize=9)

    # Title and Labels (Matching plain style)
    ax.set_title("LSTM Performance Comparison: Scaled MAE vs. Sequence Length", fontsize=12)
    ax.set_xlabel("Sequence Length", fontsize=10)
    ax.set_ylabel("Best Scaled MAE", fontsize=10)
    
    ax.set_xticks(sorted_seq_lengths)
    
    # Grid: solid light gray lines matching other plots
    ax.grid(True, linestyle='-', alpha=0.3, color='lightgray')
    
    # Keep all spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    # Adjust Y-axis limits to make room for annotations at the top
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min * 0.9, y_max * 1.15) # Add 15% head room
    
    # Legend with frame (standard style)
    ax.legend(frameon=True, loc='upper right', fontsize=10, facecolor='white', edgecolor='lightgray')

    # Ensure white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Save with enough padding to prevent clipping
    plt.tight_layout(pad=3.0) 
    output_name = json_path.replace('.json', '_comparison.png')
    plt.savefig(output_name, bbox_inches='tight', facecolor='white')
    print(f"Graph saved as: {output_name}")
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot grid search performance comparison.')
    parser.add_argument('json_file', help='Path to the grid search results JSON file')
    args = parser.parse_args()
    
    if os.path.exists(args.json_file):
        plot_performance(args.json_file)
    else:
        # Try appending .json if not present
        if not args.json_file.endswith('.json'):
            json_file = args.json_file + '.json'
            if os.path.exists(json_file):
                plot_performance(json_file)
            else:
                print(f"Error: File {args.json_file} or {json_file} not found.")
        else:
            print(f"Error: File {args.json_file} not found.")
