import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Tracker:
    def __init__(self, exp):
        self.dir = os.path.join('runs', f"{exp}")
        os.makedirs(self.dir, exist_ok=True)

        self.history = pd.DataFrame(columns=['train loss', 'train f1', 'val loss', 'val f1'])


    def __call__(self):
        self.history.to_csv(os.path.join(self.dir, f"track.csv"), index=False)

        if np.any(self.history['train f1']):
            self._plot('f1', self.history['train f1'], self.history['val f1'], np.argmax(self.history['val f1']))
        else:
            self._plot('loss', self.history['train loss'], self.history['val loss'], np.argmin(self.history['val loss']))


    def __iadd__(self, metrics):
        self.history.loc[len(self.history)] = list(metrics)

        return self


    def _plot(self, metric, train_metric, val_metric, optimum):
        plt.figure(figsize=(10, 5))

        plt.plot(np.arange(len(train_metric)) + 1, train_metric, label=f'Train {metric}', color='b')
        plt.plot(np.arange(len(val_metric)) + 1, val_metric, label=f'Validation {metric}', color='g')

        plt.scatter(optimum + 1, val_metric[optimum], color='red', marker='x', label='Optimum')
        plt.text(optimum + 1, val_metric[optimum], f'{val_metric[optimum]:.4f}', color='red', ha='center', va='bottom')

        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.dir, f"{'pretrain-' if not np.any(self.history['train f1']) else ''}{metric}.png"))
        plt.close()
