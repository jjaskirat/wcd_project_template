class Experiment:
    def __init__(self):
        self.val_loss_list = []
        self.val_metric_list = []
        self.train_loss_list = []
        self.train_metric_list = []
        self.test_loss = None
        self.test_metric = None
        self.save_dir = f'{ROOT_DIR}/models'