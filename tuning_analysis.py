import pandas as pd
import re

# Path to your results file
results_file = 'project_files/tuning_files/July/optimization_results_20250711_165145.txt'

# Prepare storage
rows = []

with open(results_file, 'r') as f:
    lines = f.readlines()

# Keep track of current settings
current_iterations = None
current_depth = None
current_importance = None

# Regex patterns to detect section headers and rows
iter_pattern = re.compile(r'iters\s*=\s*(\d+)', re.IGNORECASE)
depth_pattern = re.compile(r'depth\s*=\s*(\d+)', re.IGNORECASE)
imp_pattern = re.compile(r'importance\s*=\s*([\d.]+)', re.IGNORECASE)
learn_pattern = re.compile(r'learning_rate\s*=\s*([\d.]+)', re.IGNORECASE)

# Loop through lines
for line in lines:
    line = line.strip()

    m_iter = iter_pattern.search(line)
    m_depth = depth_pattern.search(line)
    m_imp = imp_pattern.search(line)
    m_learn = learn_pattern.search(line)

    if m_iter:
        current_iterations = int(m_iter.group(1))
    if m_depth:
        current_depth = int(m_depth.group(1))
    if m_imp:
        current_importance = float(m_imp.group(1))
    if m_learn:
        current_learning_rate = float(m_learn.group(1))

    # Skip empty or header lines
    if not line or line.startswith('threshold') or line.startswith('Result') or line.startswith('---'):
        continue

    # Split line into parts
    parts = re.split(r'\s+', line)

    # Debug print AFTER splitting
    # if len(parts) >= 5:
    #     print('DEBUG: parts:', parts)

    # Only process lines with the expected number of numeric fields
    if len(parts) >= 13:
        try:
            threshold = float(parts[0])
            sensitivity = float(parts[1])
            specificity = float(parts[2])
            f1 = float(parts[3])
            accuracy = float(parts[4])
            net_benefit = float(parts[5])
            tp = int(parts[6])
            fp = int(parts[7])
            fn = int(parts[8])
            tn = int(parts[9])
            auc = float(parts[10])
            auprc = float(parts[11])
            ece = float(parts[12])

            rows.append({
                'Iterations': current_iterations,
                'Depth': current_depth,
                'Importance': current_importance,
                'Learning Rate': current_learning_rate,
                'Threshold': threshold,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'F1': f1,
                'Accuracy': accuracy,
                'Net Benefit': net_benefit,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn,
                'AUC': auc,
                'AUPRC': auprc,
                'ECE': ece
            })
        except Exception as e:
            print('Error parsing line:', parts, e)
            continue

# Convert to DataFrame
df = pd.DataFrame(rows)
# print('First few rows of df before grouping:\n', df.head(10))

# # For each configuration, keep the row with the highest Net Benefit
# best_rows = df.sort_values('Net Benefit', ascending=False).groupby(
#     ['Iterations', 'Depth', 'Importance']
# ).head(1).reset_index(drop=True)

# # Sort for readability
# best_rows = best_rows.sort_values(by=['Net Benefit', 'F1'], ascending=False)

# # Output summary table
# print(best_rows.to_string(index=False))

# # Optionally save to CSV
# best_rows.to_csv('Threshold_Summary.csv', index=False)



# Load your detailed CSV (all thresholds)
# df = pd.read_csv('Threshold_optimization_results.csv')

# Filter out degenerate threshold=0
df = df[df['Threshold'] > 0]

# Keep only rows where Sensitivity >= 0.8
df = df[df['Sensitivity'] >= 0.8]

# For each configuration, keep the row with the highest F1
best_rows = df.sort_values('F1', ascending=False).groupby(
    ['Iterations', 'Depth', 'Importance']
).head(1).reset_index(drop=True)

# Sort for readability
best_rows = best_rows.sort_values(by=['F1', 'Net Benefit'], ascending=False)

# Print
print(best_rows.to_string(index=False))

# Optionally save to CSV for inspection
best_rows.to_csv('Linux_July_5_Threshold_Best_Configurations.csv', index=False)
