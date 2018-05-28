class Scheduler:
    def __init__(self,lr,total_epoches):
        self._lr = lr
        self._total_epoches=total_epoches

    def get_learning_rate(self):
        return self._lr
    def get_total_epoches(self):
        return self._total_epoches