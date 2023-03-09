import numpy as np


def get_input_mode(msg):
    while True:
        input_ = input(msg)
        print('\n')
        try:
            if input_ == 't':
                mode = 'train'
                break
            elif input_ == 'p':
                mode = 'predict'
                break
            elif input_ == 'q':
                quit()
            else:
                raise ValueError
        except ValueError:
            print('Try again please.')
    return mode

def get_input_yn(msg):
    while True:
        input_ = input(msg)
        try:
            if input_ == 'n':
                quit()
            elif input_ == 'y':
                break
            else:
                raise ValueError
        except ValueError:
            print('Try again please.')
    return input_

def get_input_pred():
    while True:
        print('Enter coordinates to predict class or "q" to quit.')
        try:
            x_input = input('Enter x coordinate: ')
            if x_input == 'q':
                quit()
            else:
                x_input = float(x_input)
                y_input = input('Enter y coordinate: ')
                y_input = float(y_input)
                if x_input < -1 or x_input >= 1 \
                    or y_input < -1 or y_input >= 1:
                    raise ValueError
                else:
                    x = np.reshape([x_input, y_input], (2, 1))
                    return x
        except ValueError:
            print('\nERROR: bad input, Try again please. Datapoint should be in the range between but not uncluding -1 and 1.\n')
            continue
        