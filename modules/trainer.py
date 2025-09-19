import os
import numpy as np
import torch
import torch.nn.functional as F
from modules.meter import Meter
from modules.tracker import Tracker

def f1_score(logits, labels):

    predicted = logits.argmax(1)
    num_classes = logits.size(1)
    f1_scores = []

    for c in range(num_classes):
        true_positives = ((predicted == c) & (labels == c)).sum().item()
        predicted_positives = (predicted == c).sum().item()
        actual_positives = (labels == c).sum().item()

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        recall = true_positives / actual_positives if actual_positives > 0 else 0.0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / num_classes

class Trainer:
    def __init__(self, model, exp, logger):
        self.model = model
        self.logger = logger
        self.tracker = Tracker(exp)


    def __call__(self, train_loader, val_loader, criterion, epochs, lr, weight_decay=1e-3, catchup_epochs=10):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        max_f1 = -np.inf
        min_loss = np.inf

        for epoch in range(epochs):
            self.logger(f"Epoch {epoch+1}/{epochs}:")

            self.model.set_trainable(False) if epoch < catchup_epochs else self.model.set_trainable(True)

            train_loss, train_f1 = self._propagate(train_loader, criterion, optimizer, back_prop=True)
            self.logger(f"\tTrain Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")

            with torch.no_grad():
                val_loss, val_f1 = self._propagate(val_loader, criterion, optimizer, back_prop=False)
            self.logger(f"\tValidation Loss: {val_loss:.4f}, Validation F1: {val_f1:.4f}")

            if criterion == F.cross_entropy and val_f1 > max_f1:
                self.model.save()
                self.logger(f"Model saved to {os.path.join(self.model.dir, 'model.pth')}")

                max_f1 = val_f1
            elif criterion == F.triplet_margin_loss and val_loss < min_loss:
                self.model.save()
                self.logger(f"Model saved to {os.path.join(self.model.dir, 'model.pth')}")

                min_loss = val_loss

            self.tracker += (train_loss, train_f1, val_loss, val_f1)

        self.logger(f"Final F1: {max_f1:.4f}")

        self.tracker()

        return max_f1


    def _propagate(self, data_loader, criterion, optimizer, back_prop):
        self.model.train() if back_prop else self.model.eval()

        loss_meter = Meter()
        f1_meter = Meter()

        miner = TripletMarginMiner(margin=0.2, type_of_triplets='semihard')

        for i, (*inputs, targets, modes) in enumerate(data_loader):
            self.logger.tqdm(i + 1, len(data_loader))

            logits = self.model(*inputs)

            if criterion == F.triplet_margin_loss:
                (a, p, n) =  miner(F.normalize(logits), targets)

                loss = criterion(logits[a], logits[p], logits[n], margin=0.2)
                f1 = np.nan
            else:
                loss = criterion(logits, targets)
                f1 = f1_score(logits, targets)

            f1_meter(f1, targets.size(0))
            loss_meter(loss.item(), targets.size(0))

            if back_prop:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss_meter.avg, f1_meter.avg
