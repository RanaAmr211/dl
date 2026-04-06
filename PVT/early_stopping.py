"""
Early Stopping for PVT-Tiny only
"""
class PVTEarlyStopping:
    def __init__(self, patience=5, verbose=True):
        """
        Args:
            patience: Number of epochs to wait if no improvement
            verbose: Print messages
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc, epoch):
        """Check if should stop training"""
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f'✓ Val Acc improved to {score:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f' No improvement. Counter {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping at epoch {epoch}')

        return self.early_stop