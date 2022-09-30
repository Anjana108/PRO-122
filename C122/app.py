import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split as spl
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score as ac 
import cv2

x = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]

print(pd.Series(y).value_counts())

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
lenclass = len(classes)

sample_count = 5
fig = plt.figure(figsize=(lenclass*2, (1+sample_count*2)))

idx = 0
for cls in classes:
    ind = np.flatnonzero(y==cls)
    ind = np.random.choice(ind, sample_count, replace = False)

    i = 0

    for idx in ind:
        plt_ind = 1*lenclass + idx + 1
        p = plt.subplot(sample_count, lenclass, plt_ind)
        p = sns.heatmap(np.reshape(x[idx], (22, 30)), cmap = plt.cm.gray, xticklabels=False, yticklabels=False, cbar=False)
        p = plt.axis('off')

        i += 1
    idx += 1

x_train, x_test, y_train, y_test = spl(x, y, random_state=9, train_size=7500, test_size=2500)

x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

clf = lr(solver='saga', multi_class='multinomial').fit(x_train_scaled,  y_train)

y_pred = clf.predict(x_test_scaled)

accuracy = ac(y_test, y_pred)
print(accuracy)

cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
p = plt.figure(figsize=(10, 10))
p = sns.heatmap(cm, annot=True, fmt = "d", cbar = False)