import matplotlib.pyplot as plt
import numpy as np

from process import process_forward

def plot_data(classes, X_train, Y_train, X_test, Y_test):
    fig, axs = plt.subplots(1, 2, sharey=False)
    n = classes
    ax1 = axs[0].scatter(X_train[:, 0], y = X_train[:, 1], c=Y_train, s=80, cmap=discrete_cmap(n, 'rainbow'))
    ax2 = axs[1].scatter(X_test[:, 0], y = X_test[:, 1], c=Y_test, s=80, cmap=discrete_cmap(n, 'rainbow')) 
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')
    cbar = fig.colorbar(ax2, ax=axs, ticks=range(0,n), shrink=0.7)
    #cbar = fig.colorbar(ax2, ax=axs, ticks=range(0,n))
    cbar.set_label('Classes')
    ax1.set_clim(-0.5, n - 0.5)
    ax2.set_clim(-0.5, n - 0.5)
    axs[0].set_xlim(-1.0, 1.0)
    axs[1].set_xlim(-1.0, 1.0)
    axs[0].set_ylim(-1.0, 1.0)
    axs[1].set_ylim(-1.0, 1.0)
    axs[0].set_title('Training Data')
    axs[1].set_title('Test Data')
    plt.show()


def plot_training(points, error_points, epochs, error_epochs):
    fig, axs = plt.subplots(1, 2, sharey=False)
    axs[0].plot(points, error_points)
    axs[1].plot(epochs, error_epochs)
    axs[0].set_title('Error for each Datapoint\n(Mean Squared Error for CalcTanh, Binary Cross-Entropy for CalcSoftmax)')
    axs[1].set_title('Mean Error over Complete Dataset')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Error')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Error')
    plt.show()


def create_mesh(h, network):      
    x_coord_min, x_coord_max = -1, 1
    y_coord_min, y_coord_max = - 1, 1
    x_coord_mesh = np.arange(x_coord_min, x_coord_max, h)
    y_coord_mesh = np.arange(y_coord_min, y_coord_max, h)
    x_coord_mesh, y_coord_mesh = np.meshgrid(x_coord_mesh, y_coord_mesh)
    pred_res_mesh = []
    pred_prob_mesh = []
    for x_coord_row, y_coord_row in zip(x_coord_mesh, y_coord_mesh):
        pred_res_mesh_row = []
        pred_prob_mesh_row = []
        for x_coord, y_coord in zip(x_coord_row, y_coord_row):
            x = np.reshape([x_coord, y_coord], (2, 1))
            y_pred, pred_res, pred_prob = process_forward(x, network)
            pred_res_mesh_row.append(pred_res)
            #pred_prob_mesh_row.append(pred_prob)
        pred_res_mesh.append(pred_res_mesh_row)
        #pred_prob_mesh.append(pred_prob_mesh_row)
    pred_res_mesh_r = np.reshape(pred_res_mesh, (len(x_coord_mesh), len(y_coord_mesh)))
    #pred_prob_mesh_r = np.reshape(pred_prob_mesh, len(x_coord_mesh)*len(y_coord_mesh))
    return x_coord_mesh, y_coord_mesh, pred_res_mesh_r


def plot_testing(classes, X_train, Y_train, X_test, pred_res, pred_prob, x_coord_mesh, y_coord_mesh, pred_res_mesh_r):
    n = classes
    fig, axs = plt.subplots(1, 2, sharey=False)
    ax1 = axs[0].scatter(X_train[:, 0], X_train[:, 1], 
            c=Y_train, 
            s=80, 
            edgecolors='none', 
            cmap=discrete_cmap(n, 'rainbow'))
    ax2 = axs[1].contourf(x_coord_mesh, y_coord_mesh, pred_res_mesh_r, cmap=discrete_cmap(n, 'rainbow'), alpha=0.2)
    ax3 = axs[1].scatter(X_test[:, 0], X_test[:, 1], 
            c=pred_res, 
            s=80, 
            alpha=pred_prob, 
            edgecolors='None', 
            cmap=discrete_cmap(n, 'rainbow'))
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')
    cbar = fig.colorbar(ax3, ax=axs, ticks=range(0,n), shrink=0.7)
    cbar.set_label('Classes')
    ax1.set_clim(-0.5, n - 0.5)
    ax2.set_clim(-0.5, n - 0.5)
    axs[0].set_xlim(-1.0, 1.0)
    axs[1].set_xlim(-1.0, 1.0)
    axs[0].set_ylim(-1.0, 1.0)
    axs[1].set_ylim(-1.0, 1.0)
    axs[0].set_title('Training Data')
    axs[1].set_title('Predicted Data\n(Transparency is anti-proportional to calculated probability of prediciton)')
    axs[0].set_label('label')
    plt.show()

    
def plot_prediction(x_coord_mesh, y_coord_mesh, pred_res_mesh_r, x_input, pred_res, pred_prob):    
    fig, ax = plt.subplots()
    n = len(np.unique(pred_res_mesh_r))    
    contour = ax.contourf(x_coord_mesh, y_coord_mesh, pred_res_mesh_r, cmap=discrete_cmap(n, 'rainbow'))
    #contour = ax.contourf(x_coord_mesh, y_coord_mesh, pred_res_mesh_r, cmap=discrete_cmap(n, 'rainbow'), alpha=pred_prob_mesh_r)      
    ax.scatter(x_input[0], x_input[1], c=pred_res, s=80)
    ax.set_aspect('equal')
    ax.annotate(f'Predicted Class is {pred_res},\nProbability of Prediction is {pred_prob})', 
                xy=(x_input[0], x_input[1]), 
                xytext=(6, 6), 
                textcoords="offset pixels") 
    cbar = plt.colorbar(contour, ax=ax, ticks=range(0,n), shrink=1, spacing='proportional')
    cbar.set_label('Classes')
    ax.set_title('Decision Boundaries and Prediction Results')
    contour.set_clim(-0.5, n - 0.5)
    plt.show()
    

def discrete_cmap(N, base_cmap=None):
    # By Jake VanderPlas
    # License: BSD-style
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


