import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

"""models = ["RF", "LogReg", "SVM Linear", "SVM RBF"]
modelScores = [.943, .893, .872, .927]
dummy = .472

#ci = [(.8, .9), (.8, .9), (.8, .9), (.5, .7)]
#y_r = [modelScores[i] - ci[i][1] for i in range(len(ci))]

#plt.bar(np.arange(len(models)), modelScores, align='center', alpha=0.5, yerr=y_r)
plt.bar(np.arange(len(models)), modelScores, align='center', alpha=0.5)
plt.xticks(np.arange(len(models)), models)
plt.title('F1 Scores')
plt.ylabel('F1 Score')
plt.xlabel('Models')
plt.plot([-.5, 5], [dummy, dummy], "k--")
plt.show()"""

models = ["Dummy", "RF", "Log Reg", "SVM Linear", "SVM RBF"]
metrics = ["Accuracy", "Precision", "F1-Score", "Recall", "ROC AUC"]
#models = models * 5
colors = ['red', 'blue', 'yellow', 'green', 'orange', 'orange']
colors = colors * 5
print len(metrics)
modelScores = [.618, .944, .893, .871, .927, 0,
 .381, .946, .893, .874, .927, 0,
 .472, .943, .893, .893, .872, 0,
 .618, .944, .893, .871, .927, 0,
 .5, .931, .889, .870, .923]
print len(modelScores)

plt.bar(np.arange(len(modelScores)), modelScores, align='center', alpha=0.5, color=colors)
plt.xticks([2, 8, 14, 20, 26], metrics)
plt.title('Model Performance')
plt.ylabel('Scores')
plt.xlabel('Metrics')
patches = []
for c in range(len(models)):
	patch = mpatches.Patch(color=colors[c], label=models[c])
	patches.append(patch)
plt.legend(handles=patches)
plt.show()