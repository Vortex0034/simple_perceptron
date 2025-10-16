import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Layer:

    def __init__(self, input_feat_count, output_count, layer_number, random_s=1, is_first=False, is_last=False):
        self.layer_number = layer_number
        self._last_input = None
        self.is_last = is_last
        self.probab = None
        self.is_first = is_first
        self._weights = None
        self._zero_weights = None
        self._last_activation = None
        state = np.random.RandomState(random_s)

        # строки индексируются по входным признакам, столбцы - по текущим нейронам
        self._weights = state.rand(input_feat_count, output_count)
        self._zero_weights = state.rand(1, output_count)

    def update_weights(self, eta, y=None, last_loss_function_delta=None, last_loss_weights=None):
        #обновляет веса, взвращает массив для рекуррентного вычисления на следующем шаге и старые веса
        current_eta = (eta*(10**(3-self.layer_number)))
        delta = None
        delta_b = None
        last_delta_weights_dot = None
        transpose_activation = self._last_input.T
        if not self.is_last:
            last_delta_weights_dot = last_loss_function_delta.dot(last_loss_weights.T) * self.sig_activation_diff()
        else:
            last_delta_weights_dot = (self._last_activation - y)
        delta = transpose_activation.dot(last_delta_weights_dot)
        delta_b = np.ones((1, transpose_activation.shape[1])).dot(last_delta_weights_dot)
        self._weights -= current_eta*delta
        self._zero_weights -= current_eta*delta_b
        return last_delta_weights_dot, self._weights

    def predict(self, X):
        self._last_input = X
        self.probab = self.sig_activation(X)
        result = None
        if self.is_last:
            mask = self.probab.max(axis=1,keepdims=1) == self.probab
            result = np.where(mask, 1, 0)
        else:
            result = self.probab
        return result

    def vector_scalar(self, X):
        # выход: строки - выходы по нейронам по каждому образцу (N X this_layer_n)
        return X.dot(self._weights) + self._zero_weights

    def sig_activation(self, X):
        self._last_activation = 1 / (1 + np.exp(-self.vector_scalar(X)))
        return self._last_activation

    def sig_activation_diff(self):
        return self._last_activation * (1 - self._last_activation)

class Perception:

    def __init__(self, eta, layers_counts, iters, start_random_state=1):
        self.eta = eta
        self.costs = []
        self.layers_len = len(layers_counts)
        self.layers = []
        if self.layers_len > 1:
            self.layers = [Layer(layers_counts[0], layers_counts[1], 0, start_random_state, is_first=True)]
        self.iters = iters
        for i in range(1, self.layers_len - 1):
            self.layers.append(Layer(layers_counts[i], layers_counts[i+1], i, start_random_state+i, False))
        self.layers.append(Layer(layers_counts[self.layers_len-1], layers_counts[self.layers_len-1], self.layers_len - 1, start_random_state, is_last=True))

    def fit(self, X, y):
        for i in range(self.iters):
            pred = self.predict(X)
            last_delta, last_weights = self.layers[self.layers_len-1].update_weights(self.eta, y=y)
            self.costs.append(((pred != y).sum() / 2))

            for j in np.arange(self.layers_len-2, 0, -1):
                last_delta, last_weights = self.layers[j].update_weights(self.eta, last_loss_function_delta=last_delta, last_loss_weights=last_weights)

            if self.layers_len-2 >= 0:
                self.layers[0].update_weights(self.eta, last_loss_function_delta=last_delta, last_loss_weights=last_weights)

            #print(self.costs)

    def predict(self, X):
        next_output = X
        for i in range(self.layers_len):
            next_output = self.layers[i].predict(next_output)

        return next_output

    def calculate_cost(self, predicted, y):
        return predicted - y


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
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
    ppn = Perception(0.001, [11,7,7,2], 3000) # скорость обучения - 0.001, 4 слоя, 3000 эпох
    ppn.fit(X_train, y_train)
    p = ppn.predict(X_test)
    
    plt.plot(list(range(len(ppn.costs))), ppn.costs)
    plt.xlabel("Номер эпохи")
    plt.ylabel("Количество ошибок при обучении")
    print(f'Ошибок в предсказании:{((p[:,1] != y_test[:,1]).sum())} (из {len(test_samp)})')
    plt.show()