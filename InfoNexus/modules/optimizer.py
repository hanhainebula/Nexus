import torch.optim as optim


def get_optimizer(name, params, lr, weight_decay):
        r"""Return optimizer for specific parameters.

        .. note::
            If no learner is assigned in the configuration file, then ``Adam`` will be used.

        Args:
            params: the parameters to be optimized.

        Returns:
            torch.optim.optimizer: optimizer according to the config.
        """
        learning_rate = lr
        decay = weight_decay
        if name.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
        else:
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

def get_lr_scheduler():
    return None