import pandas as pd
import sys
import glob

path = sys.argv[1]

files = glob.glob(path)
dataframes = []

for f in files:
    if "averaged" in f or "latest" in f:
        continue  # Skip already averaged files
    print(f"Processing file: {f}")
    df = pd.read_csv(f)
    model = df['model'][0]
    framework = df['framework'][0]
    # Ensure timestamp is a datetime or a numerical index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    
    # Resample to 500ms intervals and take the mean of values in that bucket
    resampled = df.resample('500ms').agg({
        'model': 'first',
        'framework': 'first',
        'fps': 'mean'
    })
    dataframes.append(resampled)

## 1. Combine all files vertically (stacking rows)
combined_df = pd.concat(dataframes, axis=0)

# 2. Group by the index (timestamp) to average across multiple runs
# We use .groupby(level=0) because 'timestamp' is the index
final_df = combined_df.groupby(level=0).agg({
    'model': 'first',
    'framework': 'first',
    'fps': 'mean'
})

final_df.to_csv(f"fps-{framework}-{model}-averaged.csv")
