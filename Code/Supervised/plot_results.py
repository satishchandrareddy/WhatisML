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
    plt.xlabel("Iteration")
    plt.ylabel(",".join(key_list))
    plt.title(",".join(key_list))
    plt.legend(loc="upper right")

def plot_results_linear(Xtrain,Ytrain,model):
    # plot training data and  machine learning solution
    # determine machine learning prediction
    Xtest = np.array([[np.min(Xtrain[0,:]),np.max(Xtrain[0,:])]])
    Ytest_pred = model.predict(Xtest)
    # plot results
    plt.figure()
    plt.xlabel("House Area (1000s sq ft)")
    plt.ylabel("House Price (millions $)")
    plt.title("Linear Regression")
    plt.plot(np.squeeze(Xtrain),np.squeeze(Ytrain),"bo",label="Training Data")
    plt.plot(np.squeeze(Xtest),np.squeeze(Ytest_pred),"r-",linewidth = 3, label="Machine Learning Prediction")
    plt.legend(loc = "upper left")

def plot_results_linear_animation(Xtrain,Ytrain,model):
    # plot training data and  machine learning solution
    # determine machine learning prediction
    Xtest = np.array([[np.min(Xtrain[0,:]),np.max(Xtrain[0,:])]])
    Ytest_pred = model.predict(Xtest)
    # generate plots of machine learning prediction and create container of plots
    param_list = model.get_param_list()
    fig,ax=plt.subplots()
    ax.set_xlabel("House Area (1000s sq ft)")
    ax.set_ylabel("House Price (millions $)")
    ax.set_ylim(0, 1.1*np.max(Ytrain))
    # get plot of training data
    train = ax.scatter(np.squeeze(Xtrain),np.squeeze(Ytrain), c="b", marker="o", label="Training Data")
    model.set_param(param_list[0])
    Ymodel = model.predict(Xtest)
    ml = plt.plot(np.squeeze(Xtest),np.squeeze(Ymodel),"r-",linewidth=3,label="Machine Learning Prediction")[0]
    title = ax.set_title("Linear Regression")
    ax.legend(loc="upper left")
    
    #function to update animation
    def animate(i, param_list, Xtest):
        model.set_param(param_list[i])
        Ymodel = model.predict(Xtest)
        ml.set_data(np.squeeze(Xtest), np.squeeze(Ymodel))
        title = ax.set_title(f"Linear Regression - Iteration: {i}")
        return ml, title

    n_epochs = len(param_list)
    # create animation
    ani = animation.FuncAnimation(fig, animate, fargs=[param_list, Xtest],
                               frames=n_epochs, interval=100, blit=True)
   
    # create mp4 version of animation - need to install ffmpeg 
    # look up on internet for intallation instructions
    # ani.save('LinearRegression.mp4', writer='ffmpeg')

    return ani


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
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    
    scatter = plot_scatter(Xtrain,Ytrain,nclass)
    legend = plt.legend(loc="upper left")
    title = ax.set_title("Classification")
    model.set_param(param_list[0])
    yreshape = model.predict(np.concatenate((x0reshape,x1reshape),axis=0))
    # reshape results into 2d grid and plot heatmap
    heatmap = plt.pcolormesh(x0grid,x1grid,np.reshape(yreshape,(npoints,npoints)))
    plt.colorbar()

    def animate(i):
        # predict results (concatenated x0 and x1 1-d grids to create feature matrix)
        model.set_param(param_list[i])
        yreshape = model.predict(np.concatenate((x0reshape,x1reshape),axis=0))
        # reshape results into 2d grid and plot heatmap
        heatmap = plt.pcolormesh(x0grid,x1grid,np.reshape(yreshape,(npoints,npoints)))
        title = ax.set_title(f"Classification - Iteration: {i}")
        return heatmap, title

    n_epochs = len(param_list)
    # create animation
    ani = animation.FuncAnimation(fig, animate, frames=n_epochs, 
                                interval=100, blit=True)
    # create mp4 version of animation - need to install ffmpeg 
    # look up on internet for intallation instructions
    # ani.save('classification.mp4', writer='ffmpeg')

    return ani

def plot_data_mnist(X,Y):
    # create 5x5 subplot of mnist images
    nrow = 5
    ncol = 5
    npixel_width = 28
    npixel_height = 28
    fig,ax = plt.subplots(nrow,ncol,sharex="col",sharey="row")
    fig.suptitle("Images of Sample MNIST Digits")
    idx = 0
    for row in range(nrow):
        for col in range(ncol):
            digit_image = np.flipud(np.reshape(X[:,idx],(npixel_width,npixel_height)))
            ax[row,col].pcolormesh(digit_image,cmap="Greys")
            idx +=1

def plot_results_mnist_animation(X,Y,Y_pred,nframe):
    # number of data points
    m = X.shape[1]
    # determine number of frames
    nplot = min(m,nframe)
    # set up plot
    fig, ax = plt.subplots()
    # create 1-d grids for x and and y directions
    npixel_width = 28
    npixel_height = 28
    # list of frames
    container = []
    for idx in range(nplot):
        # digit plot - need flipud or else image will be upside down
        digit_image = np.flipud(np.reshape(X[:,idx],(npixel_width,npixel_height)))
        pc = ax.pcolormesh(digit_image,cmap="Greys",animated=True)
        digit_title = ax.text(14,28.3,"Image: {0}     Actual: {1}   Predicted: {2}".format(idx,Y[0,idx],Y_pred[0,idx]),
             size=10,ha="center", animated=True)
        # append image/title to container
        container.append([pc, digit_title])

    ani = animation.ArtistAnimation(fig,container,interval=500,repeat=False,blit=False)
    # create mp4 version of animation - need to install ffmpeg 
    # look up on internet for intallation instructions
    # ani.save('mnist.mp4', writer='ffmpeg')
    plt.show()