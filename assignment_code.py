import pandas as pd
import time
import numpy as np
import math
from sklearn.metrics import hinge_loss

class data():
    """Creating a datset that contains the train, ideal and test function.This can be considered as the parent class"""
    def __init__(self, train_data, ideal_data, test_data):
        self.train_data = train_data
        self.ideal_data = ideal_data
        self.test_data = test_data
        self.train_y_data = self.train_data.iloc[:, 1:]
        self.ideal_y_data = self.ideal_data.iloc[:, 1:]

class Formula(data):
    """The Formula class contains the def __init__ which is the daughter class to the class data"""
    def __init__(self, train_data, ideal_data, test_data):
        super().__init__(train_data, ideal_data, test_data)
        self.train_y_data = self.train_data.iloc[:, 1:]
        #From train data, we remove the column x to be able to make further calculations
        self.ideal_y_data = self.ideal_data.iloc[:, 1:]
        #From the ideal dataset, we also remove the x column to be able to calculate only with the ys
        self.train_data_columns = self.train_y_data
        #After the columns have been removed above, the train and ideal datasets are being renamed
        self.ideal_data_columns = self.ideal_y_data
        self.train = self.train_data
        self.ideal = self.ideal_data

    def least_square(self):
        """cProfile is one of the methods used to calculate the runtime. Each function is also embedded with an estimate time functions. """
        import cProfile
        cp = cProfile.Profile()
        cp.enable()

        start_time = time.time()
        obj = {}
        indx = 0

        for h in self.train_data_columns:
            obj[indx] = (self.ideal_data_columns.subtract(self.train_data_columns[h], axis=0) ** 2).sum(axis=0)
            indx += 1

        matched = [obj[x].idxmin() for x in obj.keys()]
        print(matched)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Runtime of the code is:', elapsed_time, 'seconds')

        cp.disable()
        cp.print_stats()

    def Mean_Square_Error(self):
        import cProfile
        cp = cProfile.Profile()
        cp.enable()

        start_time = time.time()
        obj = {}
        indx = 0

        for h in self.train_data_columns:
            obj[indx] = ((abs((self.ideal_data_columns.subtract(self.train_data_columns[h], axis=0))**2).mean(axis=0))/400)
            indx += 1


        matched = [obj[x].idxmin() for x in obj.keys()]
        print(matched)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Runtime of the code is:', elapsed_time, 'seconds')

        cp.disable()
        cp.print_stats()

    def Root_Mean_Square_Error(self):
        import cProfile
        cp = cProfile.Profile()
        cp.enable()

        start_time = time.time()
        obj = {}
        indx = 0

        for h in self.train_data_columns:
            obj[indx] = ((abs((self.ideal_data_columns.subtract(self.train_data_columns[h], axis=0))**2).mean(axis=0)/400)**(1/2))
            indx += 1


        matched = [obj[x].idxmin() for x in obj.keys()]
        print(matched)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Runtime of the code is:', elapsed_time, 'seconds')

        cp.disable()
        cp.print_stats()

    def Mean_Absolute_Error(self):
        import cProfile
        cp = cProfile.Profile()
        cp.enable()

        start_time = time.time()
        obj = {}
        indx = 0

        for h in self.train_data_columns:
            obj[indx] = (abs((self.ideal_data_columns.subtract(self.train_data_columns[h], axis=0)).sum(axis=0))/400)
            indx += 1

        matched = [obj[x].idxmin() for x in obj.keys()]
        print(matched)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Runtime of the code is:', elapsed_time, 'seconds')

        cp.disable()
        cp.print_stats()


    def Mean_Absolute_Error_Percentage(self):
        import cProfile
        cp = cProfile.Profile()
        cp.enable()
        start_time = time.time()
        obj = {}
        indx = 0

        for h in self.train_data_columns:
            obj[indx] = ((abs((self.ideal_data_columns.subtract(self.train_data_columns[h], axis=0))/self.ideal_data_columns).sum(axis=0)/400))
            indx += 1


        matched = [obj[x].idxmin() for x in obj.keys()]
        print(matched)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Runtime of the code is:', elapsed_time, 'seconds')

        cp.disable()
        cp.print_stats()

    def subtraction (self,e,h):
        return e-h

    def mean_squared_loss_error(self):
        import cProfile
        cp = cProfile.Profile()
        cp.enable()

        start_time = time.time()
        obj = {}
        indx = 0

        for h in self.train_data_columns:
            obj[indx] = ((np.log(self.ideal_data_columns).subtract(np.log(self.train_data_columns[h]), axis=0)**2).sum(axis=0)/400)
            indx += 1

        matched = [obj[x].idxmin() for x in obj.keys()]
        print(matched)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Runtime of the code is:', elapsed_time, 'seconds')

        cp.disable()
        cp.print_stats()

    def root_mean_squared_loss_error(self):
        import cProfile
        cp = cProfile.Profile()
        cp.enable()

        start_time = time.time()
        obj = {}
        indx = 0

        for h in self.train_data_columns:
            obj[indx] = (((np.log10(self.ideal_data_columns).subtract(np.log10(self.train_data_columns[h]), axis=0)**2).sum(axis=0)/400)**(1/2))
            indx += 1

        matched = [obj[x].idxmin() for x in obj.keys()]
        print(matched)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Runtime of the code is:', elapsed_time, 'seconds')

        cp.disable()
        cp.print_stats()

    def Mean_Bias_Error(self):
        import cProfile
        cp = cProfile.Profile()
        cp.enable()

        start_time = time.time()
        obj = {}
        indx = 0

        for h in self.train_data_columns:
            obj[indx] = ((self.ideal_data_columns.subtract(self.train_data_columns[h], axis=0).mean(axis=0))/400)
            indx += 1

        matched = [obj[x].idxmin() for x in obj.keys()]
        print(matched)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Runtime of the code is:', elapsed_time, 'seconds')

        cp.disable()
        cp.print_stats()

    def Log_Cosh_Loss(self):
        import cProfile
        cp = cProfile.Profile()
        cp.enable()

        start_time = time.time()
        obj = {}
        indx = 0

        for h in self.train_data_columns:
            obj[indx] = (np.log(np.cosh(self.ideal_data_columns.subtract(self.train_data_columns[h], axis=0).sum(axis=0))))
            indx += 1

        matched = [obj[x].idxmin() for x in obj.keys()]
        print(matched)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Runtime of the code is:', elapsed_time, 'seconds')

        cp.disable()
        cp.print_stats()
    def Kulbach_divergence(self):
        obj = {}
        indx = 0

        for h in self.train_data_columns:
          #  recorded = (self.train_data_columns[h] * (np.log2(self.train_data_columns[h] / self.ideal_data_columns)))
            obj[indx]= (self.train_data_columns[h] * (np.log2(self.train_data_columns[h] / self.ideal_data_columns))).sum(axis=0)
            indx += 1

        matched = [obj[x].idxmin() for x in obj.keys()]
        print(matched)
       # print(recorded)

    def cross_entropy_loss (self):
        obj = {}
        indx = 0

        for h in self.train_data_columns:
            #  recorded = (self.train_data_columns[h] * (np.log2(self.train_data_columns[h] / self.ideal_data_columns)))
            obj[indx] =((np.log2(self.ideal_data_columns))*self.train_data_columns[h])
            indx += 1

        matched = [obj[x].idxmin() for x in obj.keys()]
        print(matched)


Assignment_data = Formula(pd.read_csv('train.csv'), pd.read_csv('ideal.csv'), pd.read_csv('test.csv'))
#Resulting run-time and answers are then obtained from all the methods mentioned above
Assignment_data.least_square()
Assignment_data.Mean_Square_Error()
Assignment_data.Root_Mean_Square_Error()
Assignment_data.Mean_Absolute_Error()
Assignment_data.Mean_Absolute_Error_Percentage()
Assignment_data.mean_squared_loss_error()
Assignment_data.root_mean_squared_loss_error()
Assignment_data.Mean_Bias_Error()
Assignment_data.Log_Cosh_Loss()
Assignment_data.Kulbach_divergence()
Assignment_data.cross_entropy_loss()
