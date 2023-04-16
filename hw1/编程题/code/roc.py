import pandas as pd
import matplotlib.pyplot as plt


def plot_roc(x, y):
    plt.plot(x, y)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()


def auc(x, y):
    ans = 0
    for i in range(0, len(x) - 1, 1):
        ans += 0.5 * (x[i + 1] - x[i]) * (y[i + 1] + y[i])
    return ans


df = pd.read_csv('p3.csv')
df = df.sort_values(by='output')
output = df['output'].values
label = df['label'].values
neg = list(label).count(0)
pos = list(label).count(1)
TP = 0
FP = 0
FN = pos
TN = neg
roc_list = [(0, 0)]
output = list(output)
output.reverse()
for i, j in enumerate(reversed(label)):
    if j == 1:
        TP += 1
        FN -= 1
    else:
        FP += 1
        TN -= 1
    if i < len(output) - 1 and output[i + 1] == output[i]:
        continue

    TPR = TP / (TP + FN)
    FPR = FP / (TN + FP)
    roc_list.append((FPR, TPR))

x_roc = [i for i, _ in roc_list]
y_roc = [j for _, j in roc_list]
plot_roc(x_roc, y_roc)
print(auc(x_roc, y_roc))
