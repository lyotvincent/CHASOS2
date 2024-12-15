
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

FONTSIZE = 8
# plt.figure(dpi=600, figsize=(6, 9))
plt.figure(dpi=800, figsize=(12, 4))
# plt.figure(dpi=600, figsize=(4, 4))
# plt.style.use("fast")
plt.rc("font", family="Times New Roman")
params = {"axes.titlesize": FONTSIZE,
            "legend.fontsize": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "figure.titlesize": FONTSIZE,
            "font.size": FONTSIZE}
plt.rcParams.update(params)

# leave chr22 & chrX out
print("drawing test3 leave chr22 & chrX")
methods_name = ["(1) seqfeat", "(2) seqfeat+OCRfeat", "(3) seqfeat+anchorfeat", "(4) seqfeat+anchorfeat\n+OCRfeat",\
                "(5) seqfeat+funcfeat", "(6) seqfeat+OCRfeat\n+funcfeat", "(7) seqfeat+anchorfeat\n+funcfeat", "(8) seqfeat+anchorfeat\n+OCRfeat+funcfeat"]
# acc auroc auprc
seqfeat = [0.9070, 0.9653, 0.9515]
seqfeat_ocrfeat = [0.9093, 0.9680, 0.9551]
seqfeat_anchorfeat = [0.9154, 0.9726, 0.9636]
seqfeat_ocrfeat_anchorfeat = [0.9131, 0.9739, 0.9655]
seqfeat_funcfeat = [0.9200, 0.9766, 0.9665]
seqfeat_ocrfeat_funcfeat = [0.9215, 0.9766, 0.9656]
seqfeat_anchorfeat_funcfeat = [0.9284, 0.9814, 0.9739]
seqfeat_anchorfeat_ocrfeat_funcfeat = [0.9299, 0.9819, 0.9744]
y = [seqfeat, seqfeat_ocrfeat, seqfeat_anchorfeat, seqfeat_ocrfeat_anchorfeat,\
     seqfeat_funcfeat, seqfeat_ocrfeat_funcfeat, seqfeat_anchorfeat_funcfeat, seqfeat_anchorfeat_ocrfeat_funcfeat]

# plt.subplot(2, 1, 1)
plt.subplot(1, 2, 1)
total_width, n = 0.8, 8
width = total_width / n
x = np.arange(3)
x = x - (total_width - width) / 2

c = ["#ED1C24", "#FFC90E", "#FFAEC9", "#A349A4", "#22B14C", "#00A2E8", "#B97A57", "#C3C3C3"]
for i in range(len(methods_name)):
    plt.bar(x+i*width, y[i], width=width, color=c[i], label=methods_name[i])
plt.xlim([-0.5, 2.5])
plt.ylim([0.9, 0.99])
# plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
# plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.xticks(x+width*3.5, ["Accuracy", "AUROC", "AUPRC"])
# plt.xlabel('Area Under the Receiver Operating Characteristic Curve (AUROC)')
# plt.ylabel('Area Under the Precision-Recall Curve (AUPRC)')
plt.title('(a) leave chr22 & chrX')
plt.legend(loc="best")

# plt.savefig(fname="fig1&2.svg", format="svg", bbox_inches="tight")
# plt.savefig(fname="fig3.tif", format="tif", bbox_inches="tight")
# plt.savefig(fname="fig1&2.png", format="png", bbox_inches="tight")
# plt.savefig(fname="fig1&2.eps", format="eps", bbox_inches="tight")


# plt.figure(dpi=600, figsize=(4, 4))
print("drawing test3 leave cancer cell line")
seqfeat = [0.9207, 0.9722, 0.9697]
seqfeat_ocrfeat = [0.9219, 0.9730, 0.9706]
seqfeat_anchorfeat = [0.9293, 0.9789, 0.9780]
seqfeat_ocrfeat_anchorfeat = [0.9315, 0.9799, 0.9790]
seqfeat_funcfeat = [0.9384, 0.9830, 0.9816]
seqfeat_ocrfeat_funcfeat = [0.9393, 0.9831, 0.9816]
seqfeat_anchorfeat_funcfeat = [0.9455, 0.9870, 0.9864]
seqfeat_anchorfeat_ocrfeat_funcfeat = [0.9467, 0.9871, 0.9864]
y = [seqfeat, seqfeat_ocrfeat, seqfeat_anchorfeat, seqfeat_ocrfeat_anchorfeat,\
     seqfeat_funcfeat, seqfeat_ocrfeat_funcfeat, seqfeat_anchorfeat_funcfeat, seqfeat_anchorfeat_ocrfeat_funcfeat]

# plt.subplot(2, 1, 2)
plt.subplot(1, 2, 2)
for i in range(len(methods_name)):
    plt.bar(x+i*width, y[i], width=width, color=c[i], label=methods_name[i])
plt.xlim([-0.5, 2.5])
plt.ylim([0.92, 0.99])

# plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
# plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.xticks(x+width*3.5, ["Accuracy", "AUROC", "AUPRC"])
# plt.xlabel('Area Under the Receiver Operating Characteristic Curve (AUROC)')
# plt.ylabel('Area Under the Precision-Recall Curve (AUPRC)')
plt.title("(b) leave cancer cell line")
plt.legend(loc="best")

plt.savefig(fname="fig3.tif", format="tif", bbox_inches="tight")
plt.savefig(fname="fig3.png", format="png", bbox_inches="tight")
plt.savefig(fname="fig3.svg", format="svg", bbox_inches="tight")

