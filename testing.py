import numpy as np

from process import process_forward

class Testing:
    def __init__(self, error_func):
        self.error_func = error_func

    def test(self, network, X_test_r, Y_test_rd):
            self.pred_res = [] #predicted class
            self.pred_prob = []
            self.error_epoch_tracking = np.array([]) #error per epoch
            self.error_point_tracking = np.array([])
            error_sum = 0
            for x_train, y_train in zip(X_test_r, Y_test_rd):  
                y_pred, pred_res, pred_prob = process_forward(x_train, network)
                error, error_prime = self.error_func(y_train, y_pred)
                self.pred_res.append(pred_res)
                self.pred_prob.append(pred_prob)
                self.error_point_tracking = np.append(self.error_point_tracking, error)
                error_sum += error
            self.error_test = error_sum / len(X_test_r)
            print(f'\nTest executed, error (MSE) is {np.round(self.error_test, 2)}.\n')
    