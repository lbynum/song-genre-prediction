import matplotlib.pyplot as plt
import numpy as np

models = ["SVM", "RF", "LogReg", "Dummy"]
modelScores = [.88, .87, .86, .6]

ci = [(.8, .9), (.8, .9), (.8, .9), (.5, .7)]
y_r = [modelScores[i] - ci[i][1] for i in range(len(ci))]

plt.bar(np.arange(len(models)), modelScores, align='center', alpha=0.5, yerr=y_r)
plt.xticks(np.arange(len(models)), models)
plt.title('F1 Scores')
plt.ylabel('F1 Score')
plt.xlabel('Models')
plt.plot([0., 4.5], [threshold, threshold], "k--")
plt.show()