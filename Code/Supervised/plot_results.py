# plot_results.py

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def plot_results_history(history,key_list):
    plt.figure()
    linemarker = ["r-","b-","k-","g-","c-"]
    for count in range(len(key_list)):
        epoch_list = list(range(1,len(history[key_list[count]])+1))
        plt.plot(epoch_list,history[key_list[count]],linemarker[count],label=key_list[count])
    plt.xlabel("Epoch")
    plt.ylabel(",".join(key_list))
    plt.title(",".join(key_list))
    plt.legend(loc="upper right")

def plot_results_linear(Xtrain,Ytrain,model):
    # plot training data, normal equations solution, machine learning solution
    # determine machine learning prediction
    X0 = Xtrain[0,:]
    X0min = np.min(X0)
    X0max = np.max(X0)
    Xtest = np.array([[X0min,X0max]])
    Ytest_pred = model.predict(Xtest)
    # normal equation solution
    Xb = np.concatenate((Xtrain,np.ones(Ytrain.shape)),axis=0)
    Wb = np.dot(np.dot(Ytrain,Xb.T),np.linalg.inv(np.dot(Xb,Xb.T)))
    W = Wb[0,0]
    b = Wb[0,1]
    Ynorm = W*Xtest+b
    # plot results
    plt.figure()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title("Linear Regression")
    plt.plot(np.squeeze(Xtrain),np.squeeze(Ytrain),"bo",label="Training Data")
    plt.plot(np.squeeze(Xtest),np.squeeze(Ytest_pred),"r-",linewidth = 3, label="Machine Learning Prediction")
    plt.plot(np.squeeze(Xtest),np.squeeze(Ynorm),"k-",label="Normal Equation Prediction")
    plt.legend(loc = "upper left")

def plot_results_linear_animation(Xtrain,Ytrain,model):
    X0 = Xtrain[0,:]
    X0min = np.min(X0)
    X0max = np.max(X0)
    Xtest = np.array([[X0min,X0max]])
    Ytest_pred = model.predict(Xtest)
    # normal equation solution
    Xb = np.concatenate((Xtrain,np.ones(Ytrain.shape)),axis=0)
    Wb = np.dot(np.dot(Ytrain,Xb.T),np.linalg.inv(np.dot(Xb,Xb.T)))
    W = Wb[0,0]
    b = Wb[0,1]
    Ynorm = W*Xtest+b
    # generate plots of machine learning prediction and create container of plots
    param_list = model.get_param_list()
    container = []
    fig,ax=plt.subplots()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title("Linear Regression")
     # get plot of training data
    train, = plt.plot(np.squeeze(Xtrain),np.squeeze(Ytrain),"bo",markersize=5,label="Training Data")
    # get plot of normal equations solution
    normal, = plt.plot(np.squeeze(Xtest),np.squeeze(Ynorm),"k-",linewidth=3,label="Normal Equation Prediction")
    model.set_param(param_list[0])
    Ymodel = model.predict(Xtest)
    ml, = plt.plot(np.squeeze(Xtest),np.squeeze(Ymodel),"r-",linewidth=3,label="Machine Learning Prediction")
    plt.legend(loc="upper left")
    for param in param_list:
        model.set_param(param)
        Ymodel = model.predict(Xtest)
        ml, = plt.plot(np.squeeze(Xtest),np.squeeze(Ymodel),"r-",linewidth=3,label="Machine Learning Prediction")

        list_components = [train,normal,ml]
        container.append(list_components)
    # create animation
    ani = animation.ArtistAnimation(fig,container,interval=200,repeat=False,blit=True)
    # create mp4 version of animation - need to install ffmpeg 
    # look up on internet for intallation instructions
    #ani.save('LinearRegression.mp4', writer='ffmpeg')

def plot_results_classification(Xtrain,Ytrain,model,nclass=2):
    plot_results_data(Xtrain,Ytrain,nclass)
    plot_results_heatmap(Xtrain,model)

def plot_results_data(Xtrain,Ytrain,nclass=2):
    # plot training data - loop over labels (0, 1) and points in dataset which have those labels
    plt.figure()
    plot_scatter(Xtrain,Ytrain,nclass)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.legend(loc="upper left")
    plt.title("Training Data")

def plot_scatter(Xtrain,Ytrain,nclass=2):
    symbol_train = ["ro","bo","go","co","yo"]
    container = []
    for count in range(nclass):
        idx_train = np.where(np.squeeze(np.absolute(Ytrain-count))<1e-10)
        strlabeltrain = "Y = " + str(count) + " train"
        train, = plt.plot(np.squeeze(Xtrain[0,idx_train]),np.squeeze(Xtrain[1,idx_train]),symbol_train[count],label=strlabeltrain)
        container.append(train)
    return container

def plot_results_heatmap(Xtrain,model):
    # plot heat map of model results
    x0 = Xtrain[0,:]
    x1 = Xtrain[1,:]
    npoints = 100
    # create 1d grids in x0 and x1 directions
    x0lin = np.linspace(np.min(x0),np.max(x0),npoints)
    x1lin = np.linspace(np.min(x1),np.max(x1),npoints)
    # create 2d grads for x0 and x1 and reshape into 1d grids 
    x0grid,x1grid = np.meshgrid(x0lin,x1lin)
    x0reshape = np.reshape(x0grid,(1,npoints*npoints))
    x1reshape = np.reshape(x1grid,(1,npoints*npoints))
    # predict results (concatenated x0 and x1 1-d grids to create feature matrix)
    yreshape = model.predict(np.concatenate((x0reshape,x1reshape),axis=0))
    # reshape results into 2d grid and plot heatmap
    heatmap = np.reshape(yreshape,(npoints,npoints))
    plt.pcolormesh(x0grid,x1grid,heatmap)
    plt.colorbar()
    plt.title("Data and Heatmap of Prediction Results")

def plot_results_classification_animation(Xtrain,Ytrain,model,nclass=2):
     # plot heat map info
    x0 = Xtrain[0,:]
    x1 = Xtrain[1,:]
    npoints = 100
    # create 1d grids in x0 and x1 directions
    x0lin = np.linspace(np.min(x0),np.max(x0),npoints)
    x1lin = np.linspace(np.min(x1),np.max(x1),npoints)
    # create 2d grads for x0 and x1 and reshape into 1d grids 
    x0grid,x1grid = np.meshgrid(x0lin,x1lin)
    x0reshape = np.reshape(x0grid,(1,npoints*npoints))
    x1reshape = np.reshape(x1grid,(1,npoints*npoints))
    param_list = model.get_param_list()
    fig,ax = plt.subplots()
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.title("Classification")
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    container = []
    for param in param_list:
        # plot training data
        frame = plot_scatter(Xtrain,Ytrain,nclass)
        # predict results (concatenated x0 and x1 1-d grids to create feature matrix)
        model.set_param(param)
        yreshape = model.predict(np.concatenate((x0reshape,x1reshape),axis=0))
        # reshape results into 2d grid and plot heatmap
        heatmap = plt.pcolormesh(x0grid,x1grid,np.reshape(yreshape,(npoints,npoints)))
        frame.insert(0,heatmap)
        container.append(frame)
    plt.colorbar()
    ani = animation.ArtistAnimation(fig,container,interval=100,repeat_delay=1000,blit=True)
    # create mp4 version of animation - need to install ffmpeg 
    # look up on internet for intallation instructions
    #ani.save('sample.mp4', writer='ffmpeg')