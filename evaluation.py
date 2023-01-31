import numpy as np
from texttable import Texttable


def show_confusion_matrix(Y, predictions):
    table = Texttable()
    Y = np.around(Y,decimals=0)
    predictions = np.around(predictions,decimals=0)
    classes = np.unique(Y, axis=0)
    labels = ['classes']
    for truth in classes:
        row = [truth]
        labels.append(int(truth))
        for pred in classes:
            combo = [(int(truth), int(pred)), 0] #creating a tuple with all possible combibations of true and predicted classes (0, 0), (0, 1) etc.
                                                #setting count to zero
            for y, prediction in zip(Y, predictions):#iterating over actual datapoints 
                if (y, prediction) == combo[0]:#if combination of true and predicted class matches one of possible combinations
                    combo[1] += 1 #the increase count by one
            #row.append(combo)
            row.append(combo[1])
        table.add_row(row)
    table.header(labels)
    print('Confusion Matrix')
    print(table.draw()) 
    #print('\n')   
          

def show_classification_report(Y, predictions):
    try:
        table = Texttable()
        classes = np.unique(Y, axis=0)
        for class_ in classes:
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for pred, truth in zip(predictions, Y):
                if np.array_equal(pred, truth) and np.array_equal(truth, class_): #True pos.
                    tp += 1
                elif np.array_equal(pred, truth) and not np.array_equal(truth, class_): #True neg.
                    tn += 1
                elif not np.array_equal(pred, truth) and np.array_equal(truth, class_): #False pos.
                    fp += 1
                elif not np.array_equal(pred, truth) and not np.array_equal(truth, class_): #False neg.
                    fn += 1
            precision = tp/(tp + fp)
            recall = tp/(tp + fn)
            f1_score = 2 * (precision * recall)/(precision + recall)
            row = [class_, precision, recall, f1_score]
            table.add_row(row)
        table.header(['Class', 'Precision', 'Recall', 'F1-Score'])
        print('\nClassification Report')
        print(table.draw())
        #print('\n') 
    except ZeroDivisionError:
        print('\nERROR: zero division in F1-Score calculation, classification report cannot be shown.\n')


    




 
