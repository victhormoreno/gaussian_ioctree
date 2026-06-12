import matplotlib.pyplot as plt
import csv
from datetime import datetime
import sys

def load_file(path):
    timestamps = []
    ram = []
    cpu = []

    start_time = None

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            t = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")

            if start_time is None:
                start_time = t

            # normalize time (seconds from start)
            dt = (t - start_time).total_seconds()

            timestamps.append(dt)
            ram.append(float(row["ram_mb"]))
            cpu.append(float(row["cpu_percent"]))

    return timestamps, ram, cpu

def plot_metric(all_data, metric_index, title, ylabel):
    plt.figure()

    for name, (t, ram, cpu) in all_data.items():
        values = ram if metric_index == 0 else cpu
        file_name = name.split("/")[-1]
        plt.plot(t, values, label=file_name)

    plt.title(title)
    plt.xlabel("Time (seconds from start)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

def main(files):
    all_data = {}

    for f in files:
        all_data[f] = load_file(f)

    # RAM plot
    plot_metric(
        all_data,
        metric_index=0,
        title="RAM Usage Over Time",
        ylabel="RAM (MB)"
    )

    # CPU plot
    plot_metric(
        all_data,
        metric_index=1,
        title="CPU Usage Over Time",
        ylabel="CPU (%)"
    )

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 benchmark_plot.py file1.txt file2.txt ...")
        sys.exit(1)

    main(sys.argv[1:])