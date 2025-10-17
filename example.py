from Neuron import Perception
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'data.csv')
df = pd.read_csv(file_path)
proc_of_test = 10

df = df.sample(frac=1)
n = len(df)
print(f'Размер датасета: {n}')
test_n = n * proc_of_test // 100
train_samp = df.iloc[:n - test_n, :]
test_samp = df.iloc[n-test_n:,:]

print(f'Размер тренировочной выборки: {len(train_samp)}')
print(f'Размер тестовой выборки: {len(test_samp)}')

X_train = train_samp.iloc[:,:11].values
X_test = test_samp.iloc[:, :11].values
y_train = pd.get_dummies(train_samp.iloc[:,11]).values # One-Hot Encoding
y_test = pd.get_dummies(test_samp.iloc[:,11]).values # One-Hot Encoding
ppn = Perception(0.001, [11,7,7,2], 5000) # скорость обучения - 0.001, 4 слоя, 5000 эпох
ppn.fit(X_train, y_train)
p = ppn.predict(X_test)

plt.plot(list(range(len(ppn.costs))), ppn.costs)
plt.xlabel("Номер эпохи")
plt.ylabel("Количество ошибок при обучении")
print(f'Ошибок в предсказании:{((p[:,1] != y_test[:,1]).sum())} (из {len(test_samp)})')
plt.show()