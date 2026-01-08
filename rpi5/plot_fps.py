import matplotlib.pyplot as plt
import sys
import glob
import pandas as pd

files = [
    'fps-onnx-yolov5-averaged.csv',
    'fps-onnx-yolov11-averaged.csv',
    'fps-onnx-ssdlite-averaged.csv',
    'fps-pt-yolov5-averaged.csv',
    'fps-pt-yolov11-averaged.csv',
    'fps-pt-ssdlite-averaged.csv',
    'fps-pt-fasterrcnn-averaged.csv'
]
# path = sys.argv[1]
# files = glob.glob(path)

colors = {
    'yolov5' : 'blue',
    'yolov11' : 'red',
    'fastercnn' : 'green',
    'ssdlite' : 'black',
}
line = {
    'onnx' : '-',
    'pt' : '--',
}

for file in files:
    print(f"Processing file: {file}")
    df = pd.read_csv(file)
    model = df['model'][0]
    color = colors.get(model, 'gray')
    framework = df['framework'][0]
    linestyle = line.get(framework, '-.')

    # remove the days from the timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    plt.plot(df['timestamp'], df['fps'], label=f"{model} ({framework})", color=color, linestyle=linestyle)

plt.xlabel('Time (s)')
plt.ylabel('FPS')
plt.title('FPS over Time for Different Models and Frameworks on Raspberry Pi 5')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid()
plt.show()
