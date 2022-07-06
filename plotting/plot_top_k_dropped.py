import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

# based on: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html


def grouped_bar_chart(labels, first_group, second_group, total, filename):
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, first_group, width, label="thres: 300m")
    rects2 = ax.bar(x + width / 2, second_group, width, label="thres: 600m")

    ax.set_xlabel("No. of top locations")
    ax.set_ylabel("Dropped traces")
    ax.set_title(f"Dropped traces due to top-K mapping, of total {total}")
    ax.set_xticks(x)
    ax.set_xticklabels(k)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{(height / total * 100):.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(filename)


if __name__ == '__main__':
    total_traces = 258213

    # 250m cells
    k = ["100", "200", "500", "1000"]
    dropped_thres_300 = [184339, 115690, 47159, 13331]
    dropped_thres_600 = [107196, 76205, 26453, 6277]

    grouped_bar_chart(k, dropped_thres_300, dropped_thres_600, total_traces, "top_k_dropped_250")

    # 500m cells
    k = ["100", "200", "500"]
    dropped_thres_300 = [99754, 50179, 5342]
    dropped_thres_600 = [67166, 22438, 1161]

    grouped_bar_chart(k, dropped_thres_300, dropped_thres_600, total_traces, "top_k_dropped_500")
