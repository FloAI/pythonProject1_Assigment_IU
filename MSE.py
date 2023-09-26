import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.io import show
import sqlalchemy as db
import numpy as np


class data():
    def __init__(self, train_data, ideal_data, test_data):
        self.train_data = train_data
        self.ideal_data = ideal_data
        self.test_data = test_data
        self.train_y_data = self.train_data.iloc[:, 1:]
        self.ideal_y_data = self.ideal_data.iloc[:, 1:]


class Assignment(data):
    def __init__(self, train_data, ideal_data, test_data):
        super().__init__(train_data, ideal_data, test_data)
        self.train_y_data = self.train_data.iloc[:, 1:]
        self.ideal_y_data = self.ideal_data.iloc[:, 1:]
        self.train_data_columns = self.train_y_data
        self.ideal_data_columns = self.ideal_y_data
        self.train = self.train_data
        self.ideal = self.ideal_data
        self.test = self.test_data
        self.engine = db.create_engine('sqlite:///test.db')
        #Creating an engine called testing.db
        self.connection = self.engine.connect()
        #The database connection is then established

    def Assignment(self):

        self.train_data.to_sql('train_data', self.connection, if_exists='append', index=False)
        #The train dataset is then added to the database

        results = self.connection.execute('select * from train_data ').fetchall()
        df = pd.DataFrame(results)
        df.columns = results[0].keys()
        #It can then be verified if the train datatable was successfully added to the database
        print(df)

        self.ideal_data.to_sql('ideal_data', self.connection, if_exists='append', index=False)
        result = self.connection.execute('select * from ideal_data').fetchall()
        ideal = pd.DataFrame(result)
        #The operation above is then repeated for the ideal dataset
        print(ideal)

        obj = {}
        indx = 0

        for h in self.train_data_columns:
            recorded = ((self.ideal_data_columns.subtract(self.train_data_columns[h], axis=0)) ** 2)
            #The deviation between the columns in train data and ideal data are then recorded
            obj[indx] = recorded.sum(axis=0)
            #The sum is then found with an iteration over each column in train data
            indx += 1

        max_deviation = (recorded ** (1 / 2)).max(axis=0)
        #The maximum deviation is then recordedn times square root of 2

        figures = list(obj.values())
        fig = pd.DataFrame(figures)

        matched = [obj[x].idxmin() for x in obj.keys()]
        #The individual column min names are then pulled out

        maximum_dev = (max_deviation[matched])
        #The maximum deviation is then recorded for the later use in the second criterion
        # print(maximum_dev)

        for score in matched:
            matched_data = self.ideal_data[score]

        [f, s, m, w] = [matched[0], matched[1], matched[2], matched[3]]

        new_data = ['x'] + [f, s, m, w]
        #The x column that was earlier removed from the dataset is returned.
        final_data = pd.DataFrame(new_data)
        ideal_function = self.ideal_data[new_data]
        #print(ideal_function)

        x = ideal_function.iloc[:, 0]
        y = ideal_function.iloc[:, 1]
        z = ideal_function.iloc[:, 2]
        q = ideal_function.iloc[:, 3]
        r = ideal_function.iloc[:, 4]

       #The results can then be plotted using bokeh
        output_file('ASSIGNMENT.html')
        graph = figure(title=matched[0])
        graph.scatter(x, y)
        show(graph)

        output_file('ASSIGNMEN.html')
        graph = figure(title=matched[1])
        graph.scatter(x, z)
        show(graph)

        output_file('ASSIGNM.html')
        graph = figure(title=matched[2])
        graph.scatter(x, q)
        show(graph)

        output_file('ASSIGN.html')
        graph = figure(title=matched[3])
        graph.scatter(x, r)
        show(graph)

        df_ideal_function = ideal_function.sort_values(by=['x'])
        #The values of the 4 choosen ideal functions are sorted by their values of x
        # print(df_ideal_function)

        df_test_data = self.test_data.sort_values(by=['x'])
        #The same sorting is done for the test data set to ensure they are the same with that in ideal
        t = df_test_data['x']
        #The value of column x in test data is recorded
        l = df_test_data['y']
        # The value of column y in test data is recorded

        # print(df_test_data)

        #The length of the test data and ideal datsets are then recorded

        df_test_data_len = len(df_test_data)
        df_ideal_function_len = len(df_ideal_function)

        df_test_data_new = pd.DataFrame(columns=df_test_data.columns)
        df_ideal_function_new = pd.DataFrame(columns=df_ideal_function.columns)
        idx = 0

        for i in range(0, df_test_data_len):
            df_test_data_x = df_test_data.iloc[i]['x']

            for j in range(0, df_ideal_function_len):
                df_ideal_function_x = df_ideal_function.iloc[j]['x']
       #The columns of the individual datasets are then compared to find matching value sof x and exclude the other ones
                if (df_ideal_function_x == df_test_data_x):
                    df_ideal_function_new.loc[idx] = df_ideal_function.iloc[j]
                    df_test_data_new.loc[idx] = df_test_data.iloc[i]
                    idx += 1
                    break

        ideal_y = df_ideal_function_new.iloc[:, 1:]
        #column x is again removed from the new ideal function which have been cleaned
        test_y = df_test_data_new.iloc[:, 1:]
        #The same is repeated for the new test function
        dev = ideal_y.subtract(test_y['y'], axis=0)
        # print(dev)
        delta_x = dev.max(axis=1)
        # print(delta_Y)

        #The Mean squared error is then used to find the minimum value between the deviation and maximum deviation
        delta_y = (abs((dev).subtract(maximum_dev, axis=1)))
        delta_Y = delta_y.min(axis=1)
        No_of_ideal_func = delta_y.idxmin(axis=1)
        # print(delta_Y)
        # print(No_of_ideal_func)

      #The required test table is then recorded before being transferred to the database
        tables = {'x (test func)': t, 'y (test func)': l, 'Delta y (test func)': delta_Y,
                  'No. of ideal func': No_of_ideal_func}
        table = pd.DataFrame(tables)
        table.to_sql('Thetestdatatable', self.connection, if_exists='append', index=False)
        method = self.connection.execute('select * from Thetestdatatable').fetchall()
        methods = pd.DataFrame(method)
        print(methods)

        self.connection.close()
       #The connection is then closed

Assignment_data = Assignment(pd.read_csv('train.csv'), pd.read_csv('ideal.csv'), pd.read_csv('test.csv'))
#Train, test and ideal datasets are uploaded from their csv files
Assignment_data.Assignment()
#The assignment function is then called.
