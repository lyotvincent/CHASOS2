
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FONTSIZE = 8
# plt.figure(dpi=600, figsize=(8, 4))
plt.figure(dpi=800, figsize=(4, 4))
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

methods_name = ["CHASOS_all", "CHASOS_non_functional", "CTCF-MP", "DeepCTCFLoop", "DeepLUCIA", "CharID", "Lollipop"]
auroc = [0.9819, 0.9739, 0.9779, 0.9321, 0.9617, 0.9762, 0.9678]
auprc = [0.9744, 0.9655, 0.9686, 0.9239, 0.9471, 0.9664, 0.9485]

print("drawing test1 roc prc")
# plt.subplot(1, 2, 1)
c = ["#ED1C24", "#FFC90E", "#FFAEC9", "#A349A4", "#22B14C", "#00A2E8", "#B97A57"]
for i in range(len(methods_name)):
    plt.scatter(auroc[i], auprc[i], lw=1, c=c[i], alpha=.8, label=methods_name[i])
plt.plot([min(auroc), max(auroc)], [(min(auprc)+max(auprc))/2, (min(auprc)+max(auprc))/2], linestyle='--', lw=1, color='grey', alpha=.8)
plt.plot([(min(auroc)+max(auroc))/2, (min(auroc)+max(auroc))/2], [min(auprc), max(auprc)], linestyle='--', lw=1, color='grey', alpha=.8)
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.xlabel('Area Under the Receiver Operating Characteristic Curve (AUROC)')
plt.ylabel('Area Under the Precision-Recall Curve (AUPRC)')
# plt.title('(a) Experiment 1: comparisons with a training/validation\nset partitioned based on chromosomes')
plt.legend(loc="best")

# plt.savefig(fname="fig1&2.svg", format="svg", bbox_inches="tight")
plt.savefig(fname="fig1.svg", format="svg", bbox_inches="tight")
plt.savefig(fname="fig1.tif", format="tif", bbox_inches="tight")
plt.savefig(fname="fig1.png", format="png", bbox_inches="tight")
# plt.savefig(fname="fig1&2.eps", format="eps", bbox_inches="tight")


plt.figure(dpi=800, figsize=(4, 4))
methods_name = ["CHASOS_all", "CHASOS_non_functional", "CTCF-MP", "DeepCTCFLoop", "DeepLUCIA", "CharID", "Lollipop"]
auroc = [0.9871, 0.9799, 0.9824, 0.9540, 0.9515, 0.9826, 0.9763]
auprc = [0.9864, 0.9790, 0.9802, 0.9577, 0.9525, 0.9813, 0.9707]
print("drawing test2 roc prc")
# plt.subplot(1, 2, 2)
for i in range(len(methods_name)):
    plt.scatter(auroc[i], auprc[i], lw=1, c=c[i], alpha=.8, label=methods_name[i])
plt.plot([min(auroc), max(auroc)], [(min(auprc)+max(auprc))/2, (min(auprc)+max(auprc))/2], linestyle='--', lw=1, color='grey', alpha=.8)
plt.plot([(min(auroc)+max(auroc))/2, (min(auroc)+max(auroc))/2], [min(auprc), max(auprc)], linestyle='--', lw=1, color='grey', alpha=.8)

plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.xlabel('Area Under the Receiver Operating Characteristic Curve (AUROC)')
plt.ylabel('Area Under the Precision-Recall Curve (AUPRC)')
# plt.title("(b) Experiment 2: comparisons with a training/validation\nset partitioned based on healthy/cancerous cell line")
plt.legend(loc="best")

plt.savefig(fname="fig2.svg", format="svg", bbox_inches="tight")
plt.savefig(fname="fig2.tif", format="tif", bbox_inches="tight")
plt.savefig(fname="fig2.png", format="png", bbox_inches="tight")

