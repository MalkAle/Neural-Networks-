import numpy as np
import configparser

def parser_func(section):
    #Getting data from config file
    config = configparser.ConfigParser() 
    config.sections()
    config.read('config.txt')
    define_dataset = dict(config.items(section))
    return define_dataset


def load_plot_params():
    define_plots = parser_func('define-plots')   
    try:
        plot_data_ = define_plots['plot_data']
        plot_training_ = define_plots['plot_training']
        plot_testing_ = define_plots['plot_testing']
        plot_prediction_ = define_plots['plot_prediction']
        plot_data_ = bool(plot_data_)
        plot_training_ = bool(plot_training_)
        plot_testing_ = bool(plot_testing_)
        plot_prediction_ = bool(plot_prediction_)
        return plot_data_, plot_training_, plot_testing_, plot_prediction_
    except KeyError:
        print('\nERROR: Missing input in [define-plots] section.\n' +
        'Required inputs are plot_data, plot_training, plot_testing and plot_prediction.\n')
        raise
    except ValueError:
        print("\nERROR: Ivalid input in plots, see comments in config file.\n")
        raise

def load_random_params():
    define_random_seed = parser_func('define-random')   
    try:
        random_seed = define_random_seed['random_seed']
        random_seed = bool(random_seed)
        return random_seed
    except KeyError:
        print('\nERROR: Missing input in [define-random] section.\n')
        raise
    except ValueError:
        print("\nERROR: Ivalid input in random, see comments in config file.\n")
        raise

def load_data_params():
    define_dateset = parser_func('define-dataset')
    try:
        samples = define_dateset['samples']
        classes = define_dateset['classes']
        data_kind = define_dateset['data_kind']
        samples = int(samples)
        classes = int(classes)
        data_kind = str(data_kind)
        if samples < 0 or samples > 5000 \
                or classes < 0 or classes > 10 \
                or (data_kind != 'vertical' and data_kind != 'spiral'): 
            raise ValueError
        else: 
            return samples, classes, data_kind
    except KeyError:
        print('\nERROR: Missing input in [define-dataset] section.\n' +
        'Required inputs are samples, classes and data_kind.\n')
        raise
    except ValueError:
        print("\nERROR: Ivalid input in data, see comments in config file.\n")
        raise


def load_calc_params():
    define_calc = parser_func('define-calc')
    try:
        calc_kind = define_calc["calc_kind"]
        hidden_size = define_calc["hidden_size"]
        hidden_number = define_calc["hidden_number"]
        alpha = define_calc["alpha"]
        hidden_size = int(hidden_size)
        hidden_number = int(hidden_number)
        alpha = float(alpha)
        calc_kind = str(calc_kind)
        if hidden_size < 0 or hidden_size > 20 \
            or hidden_number < 0 or hidden_number > 10 \
            or alpha < 0 or alpha > 1 \
            or (calc_kind != 'CalcSoftmax' and calc_kind != 'CalcTanh'):
            raise ValueError
        else:
            return calc_kind, hidden_size, hidden_number, alpha
    except KeyError:
        print('\nERROR: Missing input in [define-calc] section.\n' +
        'Required inputs are network_kind, hidden_size, hidden_number and alpha.\n')
        raise


def load_training_params():
    define_training = parser_func('define-training')
    try:
        learning_rate = define_training["learning_rate"]
        learning_rate = float(learning_rate)
        epochs = define_training["epochs"]
        epochs = int(epochs)
        if learning_rate < 0 or learning_rate > 1 \
            or epochs < 0 or epochs > 10000:
            raise ValueError
        else:
            return learning_rate, epochs
    except ValueError:
        print("\nERROR: Ivalid input in training, see comments in config file.\n")
        raise
    except KeyError:
        print('\nERROR: Missing input in [define-training] section.\n' +
        'Required inputs are learning_rate and epochs.\n')
        raise
   