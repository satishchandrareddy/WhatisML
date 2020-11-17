# NeuralNetwork.py

from copy import deepcopy
import functions_activation
import functions_loss
import numpy as np
import onehot

class NeuralNetwork:

    def __init__(self,nfeature):
        self.nlayer = 0
        self.nfeature = nfeature
        self.info = []
        self.param = []

    def set_param(self,param):
        for count in range(self.nlayer):
            self.info[count]["param"]["W"] = param[count]["W"]
            self.info[count]["param"]["b"] = param[count]["b"]

    def save_param(self):
        param_layer = []
        for count in range(self.nlayer):
            param_layer.append(deepcopy(self.info[count]["param"]))
        self.param.append(param_layer)

    def get_param_list(self):
        return self.param

    def add_layer(self,nunit,activation):
        if self.nlayer == 0:
            nIn = self.nfeature
        else:
            nIn = self.info[self.nlayer-1]["nOut"]
        linfo = {"nIn": nIn, "nOut": nunit, "activation": activation}
        linfo["param"] = {"W": np.random.randn(nunit,nIn), "b": np.random.randn(nunit,1)}
        linfo["param_der"] = {"W": np.zeros((nunit,nIn)), "b": np.zeros((nunit,1))}
        linfo["optimizer"] = {"W": None, "b": None}
        self.info.append(linfo)
        self.nlayer += 1

    def forward_propagate(self,X):
        for layer in range(self.nlayer):
            # linear part
            if layer == 0:
                Ain = X
            else:
                Ain = self.get_A(layer-1)
            W = self.get_param(layer,"param","W")
            b = self.get_param(layer,"param","b")
            Z = np.dot(W,Ain)+b
            # activation
            self.info[layer]["A"] = functions_activation.activation(self.info[layer]["activation"],Z)

    def back_propagate(self,X,Y):
        # compute derivative of loss
        grad_A_L = functions_loss.loss_der(self.loss,self.get_A(self.nlayer-1),Y)
        for layer in range(self.nlayer-1,-1,-1):
            # multiply by derivative of A
            grad_Z_L = functions_activation.activation_der(self.info[layer]["activation"],self.get_A(layer),grad_A_L)
            # compute grad_W L and grad_b L
            self.info[layer]["param_der"]["b"] = np.sum(grad_Z_L,axis=1,keepdims=True)
            if layer > 0:
                self.info[layer]["param_der"]["W"] = np.dot(grad_Z_L,self.get_A(layer-1).T)
                grad_A_L = np.dot(self.get_param(layer,"param","W").T,grad_Z_L)
            else:
                self.info[layer]["param_der"]["W"] = np.dot(grad_Z_L,X.T)

    def compile(self,loss_fun,optimizer_object):
        self.loss = loss_fun
        # assign deepcopy of optimizer object to W and b for each layer
        for layer in range(self.nlayer):
            self.info[layer]["optimizer"]["W"] = deepcopy(optimizer_object)
            self.info[layer]["optimizer"]["b"] = deepcopy(optimizer_object)

    def get_param(self,layer,order,label):
        # layer = integer: layer number 
        # order = string: "param" or "param_der"
        # label = string: "W" or "b"
        return self.info[layer][order][label]

    def get_A(self,layer):
        return self.info[layer]["A"]

    def get_Afinal(self):
        return self.info[self.nlayer-1]["A"]

    def compute_loss(self,Y):
        return functions_loss.loss(self.loss,self.get_Afinal(),Y)

    def update_param(self):
        # Update the parameter matrices W and b for each layer in neural network
        for layer in range(self.nlayer):
            # parameter_epoch= i = parameter_epoch=i-1 + update_epoch=i-1
            # use the += operation
            self.info[layer]["param"]["W"] += self.info[layer]["optimizer"]["W"].update(self.get_param(layer,"param_der","W"))
            self.info[layer]["param"]["b"] += self.info[layer]["optimizer"]["b"].update(self.get_param(layer,"param_der","b"))

    def fit(self,X,Y,epochs,verbose=True):
        # iterate over epochs
        loss = []
        accuracy = []
        self.save_param()
        for epoch in range(epochs):
            self.forward_propagate(X)
            self.back_propagate(X,Y)
            self.update_param()
            Y_pred = self.predict(X)
            loss.append(self.compute_loss(Y))
            accuracy.append(self.accuracy(Y,Y_pred))
            if verbose:
                print("Iteration: {} - Loss: {} - Accuracy: {}".format(epoch+1,loss[epoch],accuracy[epoch]))
            self.save_param()
        return {"loss":loss,"accuracy":accuracy}

    def predict(self,X):
        self.forward_propagate(X)
        if self.loss=="meansquarederror":
            return self.get_Afinal()
        elif self.loss=="binarycrossentropy":
            return np.round(self.get_Afinal(),0)
        elif self.loss=="crossentropy":
            return onehot.onehot_inverse(self.get_Afinal())

    def accuracy(self,Y,Y_pred):
        if self.loss == "meansquarederror":
            return np.mean(np.absolute(Y - Y_pred))
        elif self.loss == "binarycrossentropy":
            return np.mean(np.absolute(Y-Y_pred)<1e-7)
        elif self.loss == "crossentropy":
            return np.mean(np.absolute(Y-Y_pred)<1e-7)

    def summary(self):
        print(" ")
        print("Layer\tUnits In\tUnits Out\tParameters")
        nparameter_total = 0
        for layer in range(self.nlayer):
            nparameter = (self.info[layer]["nIn"]+1)*self.info[layer]["nOut"]
            nparameter_total += nparameter
            print("{}\t{}\t\t{}\t\t{}".format(layer+1,self.info[layer]["nIn"],self.info[layer]["nOut"],nparameter))
        print
        print("Total parameters: {}".format(nparameter_total))
        print(" ")