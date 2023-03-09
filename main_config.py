import numpy as np
#import matplotlib.pyplot as plt

from data import create_data
from pre_process import reshape, dummy, train_test_split
from error import mse, cce
from testing import Testing
from evaluation import show_confusion_matrix, show_classification_report
from calc import create_calc
from load_network import load_param, load_network
from plot import plot_data, plot_training, plot_testing, create_mesh, plot_prediction
from get_input import get_input_mode, get_input_yn, get_input_pred
from load_params import load_plot_params, load_random_params, load_data_params, load_calc_params, load_training_params
from process import process_forward
        

if __name__ == '__main__':
    #Intro
    print('\n----------------This is a neural network script by Alexander Malkov-----------------\n')

    #This defines with plots will be shown, reads from config.txt
    plot_data_, plot_training_, plot_testing_, plot_prediction_ = load_plot_params()

    #This defines wether random seed is used
    random_seed = load_random_params()#reads from config.txt
    if not random_seed:
        np.random.seed(10)

    #User input, continues execution skips to predict section 
    mode = get_input_mode('Train, predict or quit? (t/p/q)? ')
    while True:
        if mode == 'train':
            samples, classes, data_kind = load_data_params()#reads from config.txt
            X, Y = create_data(samples, classes, data_kind)
            
            #Train-Test-Split
            X_train, Y_train, X_test, Y_test = train_test_split(X, Y, 0.7)
            print('Train and test data created (70% train, 30% test).')
            print(f'Shapes of X_train, Y_train, X_test, Y_test are {X_train.shape}, {Y_train.shape}, {X_test.shape}, {Y_test.shape}.')
            
            #Reshaping
            X_train_r = reshape(X_train)
            X_test_r = reshape(X_test)
            Y_train_r = reshape(Y_train)
            Y_test_r = reshape(Y_test)
            print('\nData reshaped (each datapoint is passed as a vector to the neural network).')
            print(f'Shapes of X_train, Y_train, X_test, Y_test are {X_train_r.shape}, {Y_train_r.shape}, {X_test_r.shape}, {Y_test_r.shape}.')

            #Creating dummy variables for Y data
            Y_train_rd = dummy(Y_train_r)
            Y_test_rd = dummy(Y_test_r)
            print('\nDummy variables created for Y_train and Y_test.')
            print(f'Shapes of X_train, Y_train, X_test, Y_test after reshape() and dummy() are'
                    f'{X_train_r.shape},{Y_train_rd.shape},{X_test_r.shape},{Y_test_rd.shape}.\n')
            print(100*'-')
            
            #Plotting training data and test data 
            if plot_data_:
                plot_data(classes, X_train, Y_train, X_test, Y_test)
            
            #User input
            get_input_yn('Continue with creating network (y/n)? ') 
                
            #Creating Network and defining Training 
            calc_kind, hidden_size, hidden_number, alpha = load_calc_params()#reads from config.txt
            calc = create_calc(calc_kind, hidden_size, hidden_number, alpha)
            
            #Creating Training
            learning_rate, epochs = load_training_params()#reads from config.txt
            calc.create_network(input_size=X_train_r.shape[1], output_size=Y_test_rd.shape[1])
            calc.create_training(learning_rate, epochs)
            print(100*'-')

            #User input
            get_input_yn('Continue with training (y/n)? ')
            
            #Train Network
            calc.calc_train(X_train_r, Y_train_rd)

            #Plot training result
            if plot_training_:
                plot_training(points=np.arange(0,calc.training.epochs*len(X_train),1), 
                            error_points=calc.training.error_point_tracking, 
                            epochs=np.arange(0,calc.training.epochs,1), 
                            error_epochs=calc.training.error_epoch_tracking)
            
            #Saving Trained Network to File (model.npy)
            calc.save() #This saves the trained network in the 'model.npy' file, output is a list of dictionaries for each layer of network
            print(100*'-')
            
            #User input
            get_input_yn('Continue with testing (y/n)? ')

            #Testing Data
            testing = Testing(mse)
            testing.test(calc.network.network, X_test_r, Y_test_rd)

            #Evaluating Model
            show_confusion_matrix(Y_test, testing.pred_res)
            show_classification_report(Y_test, testing.pred_res)
            print(100*'-')

            #Plotting 
            if plot_testing_:
                x_coord_mesh, y_coord_mesh, pred_res_mesh_r = create_mesh(h=0.01, network=calc.network.network)
                plot_testing(classes=classes, 
                            X_train=X_train, 
                            Y_train=Y_train, 
                            X_test=X_test, 
                            pred_res=testing.pred_res, 
                            pred_prob=testing.pred_prob,
                            x_coord_mesh=x_coord_mesh,
                            y_coord_mesh=y_coord_mesh,
                            pred_res_mesh_r=pred_res_mesh_r)    

            #Get user input
            #mode = get_input_mode('Train, predict or quit? (t/p/q)? ')
            get_input_yn('Continue with prediction (y/n)? ') 
            mode = 'predict'
        
        if mode == 'predict':
            #Loading Network from File
            param = load_param() #loading parameters dictionary
            network = load_network(param) #creating the network from loaded dictionary

            while True:
                x_input = get_input_pred()
                y_pred, pred_res, pred_prob = process_forward(x_input, network)
                
                #Plotting decision boundaries and predicted datapoint
                if plot_prediction_:
                    x_coord_mesh, y_coord_mesh, pred_res_mesh_r = create_mesh(h=0.01, network=network)
                    plot_prediction(x_coord_mesh, y_coord_mesh, pred_res_mesh_r, x_input, pred_res, pred_prob)
                
        
       


        

    
    


