#!/usr/bin/env/ python
# ECBM E4040 Fall 2022 Assignment 2
# Optimizer implementations

import numpy as np


class Optimizer(object):
    def train(self, model, X_train, y_train, X_valid, y_valid,
              num_epoch=10, batch_size=500, learning_rate=1e-3, learning_decay=0.95, verbose=False, record_interval=10):

        """
        This function is for training

        Inputs:
        :param model: (class MLP) a MLP model
        :param X_train: (float32) input data, a tensor with shape (N, D1, D2, ...)
        :param y_train: (int) label data for classification, a 1D array of length N
        :param X_valid: (float32) input data, a tensor with shape (num_valid, D1, D2, ...)
        :param y_valid: (int) label data for classification, a 1D array of length num_valid
        :param num_epoch: (int) the number of training epochs
        :param batch_size: (int) the size of a single batch for training
        :param learning_rate: (float)
        :param learning_decay: (float) reduce learning rate every epoch
        :param verbose: (boolean) whether report training process
        """
        num_train = X_train.shape[0]
        num_batch = num_train // batch_size
        print('number of batches for training: {}'.format(num_batch))
        loss_hist = []
        train_acc_hist = []
        valid_acc_hist = []
        loss = 0.0
        for e in range(num_epoch):
            # Train stage
            for i in range(num_batch):
                # Order selection
                X_batch = X_train[i * batch_size:(i + 1) * batch_size]
                y_batch = y_train[i * batch_size:(i + 1) * batch_size]
                # loss
                preds = model.forward(X_batch)
                loss += model.loss(preds, y_batch)
                # update model
                self.step(model, learning_rate=learning_rate)

                if (i + 1) % record_interval == 0:
                    loss /= record_interval
                    loss_hist.append(loss)
                    if verbose:
                        print('{}/{} loss: {}'.format(batch_size * (i + 1), num_train, loss))
                    loss = 0.0

            # Validation stage
            train_acc = model.check_accuracy(X_train, y_train)
            val_acc = model.check_accuracy(X_valid, y_valid)
            train_acc_hist.append(train_acc)
            valid_acc_hist.append(val_acc)
            # Shrink learning_rate
            learning_rate *= learning_decay
            print('epoch {}: valid acc = {}, new learning rate = {}'.format(e + 1, val_acc, learning_rate))

        # Save loss and accuracy history
        self.loss_hist = loss_hist
        self.train_acc_hist = train_acc_hist
        self.valid_acc_hist = valid_acc_hist

        return loss_hist, train_acc_hist, valid_acc_hist

    def test(self, model, X_test, y_test, batch_size=10000):
        """
        Inputs:
        :param model: (class MLP) a MLP model
        :param X_test: (float) a tensor of shape (N, D1, D2, ...)
        :param y_test: (int) an array of length N
        :param batch_size: (int) seperate input data into several batches
        """
        acc = 0.0
        num_test = X_test.shape[0]

        if num_test <= batch_size:
            acc = model.check_accuracy(X_test, y_test)
            print('accuracy in a small test set: {}'.format(acc))
            return acc

        num_batch = num_test // batch_size
        for i in range(num_batch):
            X_batch = X_test[i * batch_size:(i + 1) * batch_size]
            y_batch = y_test[i * batch_size:(i + 1) * batch_size]
            acc += batch_size * model.check_accuracy(X_batch, y_batch)

        X_batch = X_test[num_batch * batch_size:]
        y_batch = y_test[num_batch * batch_size:]
        if X_batch.shape[0] > 0:
            acc += X_batch.shape[0] * model.check_accuracy(X_batch, y_batch)

        acc /= num_test
        print('test accuracy: {}'.format(acc))
        return acc

    def step(self, model, learning_rate):
        pass


class SGDOptim(Optimizer):
    def __init__(self):
        pass

    def step(self, model, learning_rate):
        """
        Implement a one-step SGD update on network's parameters
        
        Inputs:
        :param model: a neural network class object
        :param learning_rate: (float)
        """

        # get all parameters and their gradients
        params = model.params
        grads = model.grads

        for k in grads:
            ## update each parameter
            params[k] -= learning_rate * grads[k]

            
class SGDmomentumOptim(Optimizer):
    def __init__(self, model, momentum=0.5):
        """
        Inputs:
        :param model: a neural netowrk class object
        :param momentum: (float) momentum decay factor
        """
        self.momentum = momentum
        velocities = dict()
        for k, v in model.params.items():
            velocities[k] = np.zeros_like(v)
        self.velocities = velocities

    def step(self, model, learning_rate):
        """
        Implement a one-step SGD+momentum update on network's parameters
        
        Inputs:
        :param model: a neural network class object
        :param learning_rate: (float)
        """
        momentum = self.momentum
        velocities = self.velocities
        # get all parameters and their gradients
        params = model.params
        grads = model.grads
        ###################################################
        # TODO: SGD+Momentum, Update params and velocities#
        ###################################################
        #raise NotImplementedError
        for k in grads:
            velocities[k] = momentum*velocities[k] + learning_rate * grads[k]
            params[k] = params[k] - velocities[k]
        ###################################################
        #               END OF YOUR CODE                  #
        ###################################################


class AdamOptim(Optimizer):
    def __init__(self, model, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Inputs:
        :param model: a neural network class object
        :param beta1: (float) should be close to 1
        :param beta2: (float) similar to beta1
        :param eps: (float) in different case, the good value for eps will be different
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        momentums = dict()
        velocities = dict()
        for k, v in model.params.items():
            momentums[k] = np.zeros_like(v)
            velocities[k] = np.zeros_like(v)
        self.momentums = momentums
        self.velocities = velocities
        self.t = 0

    def step(self, model, learning_rate):
        """
        Implement a one-step Adam update on network's parameters
        
        Inputs:
        :param model: a neural network class object
        :param learning_rate: (float)
        """
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.eps

        momentums = self.momentums
        velocities = self.velocities
        t = self.t
        # create two new dictionaries containing all parameters and their gradients
        params, grads = model.params, model.grads
        ###################################################
        # TODO: Adam, Update t, momentums, velocities and #
        # params                                          #
        ###################################################
        #raise NotImplementedError
        for i in grads:
            t += 1
            momentums[i] = beta1*momentums[i] + (1-beta1)*grads[i]
            velocities[i] = beta2*velocities[i] + (1-beta2)*(grads[i]**2)

            a = learning_rate/((velocities[i]/(1-beta2**t))**0.5+self.eps)
            b = momentums[i]/(1-beta1**t)
            params[i] = params[i] - np.multiply(a,b)



        ###################################################
        #               END OF YOUR CODE                  #
        ###################################################

            
class NadamOptim(Optimizer):
    def __init__(self, model, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Inputs:
        :param model: a neural network class object
        :param beta1: (float) should be close to 1
        :param beta2: (float) similar to beta1
        :param eps: (float) in different case, the good value for eps will be different
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        momentums = dict()
        velocities = dict()
        for k, v in model.params.items():
            momentums[k] = np.zeros_like(v)
            velocities[k] = np.zeros_like(v)
        self.momentums = momentums
        self.velocities = velocities
        self.t = 0

    def step(self, model, learning_rate):
        """
        Implement a one-step Nadam update on network's parameters
        
        Inputs:
        :param model: a neural network class object
        :param learning_rate: (float)
        """
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.eps

        momentums = self.momentums
        velocities = self.velocities
        t = self.t
        # create two new dictionaries containing all parameters and their gradients
        params, grads = model.params, model.grads
        ###################################################
        # TODO: Nadam, Update t, momentums, velocities and#
        # params                                          #
        ###################################################
        #raise NotImplementedError
        for i in grads:
            t += 1
            momentums[i] = beta1*momentums[i] + (1-beta1)*grads[i]
            velocities[i] = beta2*velocities[i] + (1-beta2)*(grads[i]**2)
            
            a = learning_rate/((velocities[i]/(1-beta2**t))**0.5+self.eps)
            b = beta1*momentums[i]/(1-beta1**t) + (1-beta1)*grads[i]/(1-beta1**t)
            params[i] = params[i] - np.multiply(a,b)
        ###################################################
        #               END OF YOUR CODE                  #
        ###################################################


class SGDmomtraceOptim(SGDmomentumOptim):
    '''
    This class is derived from SGDmomentumOptim. 
    We will use the same SGDmomentumOptim.step function for updates. 
    Line search will be implemented by rewriting the SGDmomentumOptim.train function. 

    Please read through the code and follow the instructions below.
    '''

    def __init__(self, model, momentum=0.5, alpha1=1e-3, alpha2=0.5, max_searches=10, eps=1e-8):
        """
        Inputs:
        :param model: a neural network class object
        :param momentum: (float) momentum decay factor
        :param alpha1: (float) line search upper-bound slope
        :param alpha2: (float) line search learning rate decay factor
        :param max_searches: (int) maximum number of search loops in a single iteration
        :param eps: (float) in different case, the good value for eps will be different
        """
        super().__init__(model, momentum)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.max_searches = max_searches
        self.eps = eps

    def train(self, model, X_train, y_train, X_valid, y_valid,
              num_epoch=10, batch_size=500, learning_rate=1e-3, 
              learning_decay=0.95, verbose=False, record_interval=10):
        """
        This function is for training specifically with line search. 
        Most of the contents are replicated from Optimizer.train

        Inputs:
        :param model: (class MLP) a MLP model
        :param X_train: (float32) input data, a tensor with shape (N, D1, D2, ...)
        :param y_train: (int) label data for classification, a 1D array of length N
        :param X_valid: (float32) input data, a tensor with shape (num_valid, D1, D2, ...)
        :param y_valid: (int) label data for classification, a 1D array of length num_valid
        :param num_epoch: (int) the number of training epochs
        :param batch_size: (int) the size of a single batch for training
        :param learning_rate: (float)
        :param learning_decay: (float) reduce learning rate every epoch
        :param verbose: (boolean) whether report training process
        """

        # initialize the training
        num_train = X_train.shape[0]
        num_batch = num_train // batch_size
        print('number of batches for training: {}'.format(num_batch))
        alpha1 = self.alpha1
        alpha2 = self.alpha2
        max_searches = self.max_searches
        eps = self.eps

        # records for the entire training process
        loss_hist = []
        train_acc_hist = []
        valid_acc_hist = []

        # start epochs
        for e in range(num_epoch):
            # before starting a line search, store the initial parameters
            # for numpy arrays, remember to explicitly write .copy()
            # otherwise you would get a "view" (reference-like) instead of a real copy
            params = {k: model.params[k].copy() for k in model.params}
            grads = {k: model.grads[k].copy() for k in model.grads}
            lr = learning_rate
            num_searches = 0

            # start line search
            # we use an infinte loop here for every search step, 
            # and try to break it below
            while True:

                # local records for the current search: 
                # original loss, upper bound, loss after search
                loss, upper, loss_search = 0., 0., 0.
                # don't confuse them with the global records
                loss_local, upper_local, loss_search_local = [],[],[]
                # number of searches conducted
                num_searches += 1

                # reset the model parameters at the begining of every individual search, 
                # so that one bad trial wouldn't affect the next trial
                model.params = params
                model.grads = grads

                # iterate through the batches
                # line search is evaluated AFTER finishing all the batches
                # so just compute what is needed and accumulate the results
                # NOTE: refer to the Optimizer.train for hints of usage

                for i in range(num_batch):
                    # batch selection
                    X_batch = X_train[i * batch_size:(i + 1) * batch_size]
                    y_batch = y_train[i * batch_size:(i + 1) * batch_size]

                    # gradients & velocities for SGD + Momentum
                    grads = model.grads
                    velocities = self.velocities

                    ###################################################
                    # TODO: Calculate the loss of the model           #
                    ###################################################
                    #raise NotImplementedError
                    preds = model.forward(X_batch)
                    loss = model.loss(preds, y_batch)
                    ###################################################
                    # TODO: Update model and upper bound (RHS value)  #
                    ###################################################
                    #raise NotImplementedError
                    for k in grads:
                        velocities[k] = self.momentum*velocities[k] + learning_rate * grads[k]
                        model.params[k] = model.params[k] - velocities[k]
                        upper = loss + self.alpha1*lr*grads[k].T@model.params[k]
                    ###################################################
                    # TODO: Re-calculate the loss (LHS value)         #
                    ###################################################
                    #raise NotImplementedError
                    pred_search = model.forward(X_batch)
                    loss_search = model.loss(pred_search, y_batch)
                    ###################################################
                    #               END OF YOUR CODE                  #
                    ###################################################

                    if (i + 1) % record_interval == 0:
                        loss /= record_interval
                        upper /= record_interval
                        loss_search /= record_interval
                        loss_local.append(loss)
                        upper_local.append(upper)
                        loss_search_local.append(loss_search)
                        if verbose:
                            print('{}/{} loss: {}, upper: {}, searched: {}'.format(
                                batch_size * (i + 1), num_train, loss, upper, loss_search
                            ))
                        loss, upper, loss_search = 0., 0., 0.

                # metrics AFTER the current search:
                # the original loss (before the update)
                loss = np.mean(loss_local)
                # the upperbound
                upper = np.mean(upper_local)
                # the searched loss (after the update)
                loss_search = np.mean(loss_search_local)

                ###################################################
                # TODO: Stop searching (break) if:                #
                #           - Armijo condition is satisfied, or   #
                #           - Reached max_searches                #
                #       Else: update lr and search again          #
                ###################################################
                #raise NotImplementedError
                if loss <= loss_search or num_searches == max_searches:
                    break
                else:
                    lr = self.alpha2*lr
                ###################################################
                #               END OF YOUR CODE                  #
                ###################################################

            loss_hist += loss_local

            # Validation stage
            train_acc = model.check_accuracy(X_train, y_train)
            val_acc = model.check_accuracy(X_valid, y_valid)
            train_acc_hist.append(train_acc)
            valid_acc_hist.append(val_acc)
            # Shrink learning_rate
            learning_rate *= learning_decay
            print('epoch {}: valid acc = {}, new learning rate = {}, num searches = {}'.format(
                e + 1, val_acc, lr, num_searches
            ))

        # Save loss and accuracy history
        self.loss_hist = loss_hist
        self.train_acc_hist = train_acc_hist
        self.valid_acc_hist = valid_acc_hist

        return loss_hist, train_acc_hist, valid_acc_hist
