class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, mode, patience=7, verbose=False, delta=0., start_epoch=0, log_file=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        assert mode in ["min", "max"]
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.start_epoch = start_epoch
        self.log_file = log_file

    def __report(self):
        if self.verbose and self.counter in [round(self.patience*1/4), round(self.patience/2), round(self.patience*3/4), self.patience]:
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}. Current best score: {self.best_score}.')
            if self.log_file:
                self.log_file.write(f'EarlyStopping counter: {self.counter} out of {self.patience}. Current best score: {self.best_score}.\n')

    def __call__(self, metric, current_epoch):
        if current_epoch < self.start_epoch: return
        if self.best_score is None:
            self.best_score = metric
        elif self.mode=="min":
            if self.best_score - metric < self.delta:
                self.counter += 1
                self.__report()
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.counter = 0
            if metric < self.best_score:
                self.best_score = metric
        elif self.mode=="max":
            if metric - self.best_score < self.delta:
                self.counter += 1
                self.__report()
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.counter = 0
            if metric > self.best_score:
                self.best_score = metric