import tree
import numpy as np
import random
from threading import Thread
# from sklearn.tree import DecisionTreeRegressor


class Thread_tree(Thread):
    def __init__(self, name, new_data_x, new_data_y, i, x1, x2, x3, predictions, max_depth):
        Thread.__init__(self)
        self.name = name
        self.new_data_x = new_data_x
        self.new_data_y = new_data_y
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.i = i
        self.predictions = predictions
        self.max_depth = max_depth

    def run(self):
        '''
        Use DecisionTreeRegressor because
        faster but work as good as tree.TR
        You can check :)
        '''
        print(f'{self.name} - start')
        rtf = tree.TR(max_depth=self.max_depth)
        # rtf = DecisionTreeRegressor(max_depth=self.max_depth)
        rtf.fit(self.new_data_x[self.i], self.new_data_y[self.i])
        self.predictions[0].append(np.asarray(rtf.predict(self.x1)))
        self.predictions[1].append(np.asarray(rtf.predict(self.x2)))
        self.predictions[2].append(np.asarray(rtf.predict(self.x3)))
        print(f'{self.name} - finish')


class RandomForest:
    def __init__(self, trees=10, max_depth=10):
        self.trees = trees
        self.max_depth = max_depth

    def fit_and_predict(self, x1, x2, y1, y2, x3):
        new_data_x, new_data_y = self.shuffle(x1, y1)
        return self.predict(new_data_x, new_data_y, x1, x2, x3)

    def shuffle(self, x1, y1):
        new_data_x = []
        new_data_y = []
        for i in range(self.trees):
            tmp_x = []
            tmp_y = []
            for k in range(x1.shape[0]):
                t = random.randint(0, x1.shape[0] - 1)
                tmp_x.append(x1[t])
                tmp_y.append(y1[t])
            tmp_x, tmp_y = np.asarray(tmp_x), np.asarray(tmp_y)
            new_data_x.append(tmp_x)
            new_data_y.append(tmp_y)
        new_data_x, new_data_y = np.asarray(new_data_x), np.asarray(new_data_y)
        print(new_data_x.shape)
        return new_data_x, new_data_y

    def predict(self, new_data_x, new_data_y, x1, x2, x3):
        def create_threads(predictions):
            threads = []
            print('Just wait ~5 minutes, maybe more :) ')
            # Create threads
            for i in range(self.trees):
                name = f"Tree №{i + 1}"
                my_thread = Thread_tree(name, new_data_x, new_data_y, i, x1, x2, x3, predictions, self.max_depth)
                my_thread.start()
                threads.append(my_thread)
            # Join threads
            for t in threads:
                t.join()

        predictions = [[], [], []]
        create_threads(predictions)
        # time.sleep(5)
        prediction_1 = np.asarray(predictions[0]).mean(axis=0)
        prediction_2 = np.asarray(predictions[1]).mean(axis=0)
        prediction_3 = np.asarray(predictions[2]).mean(axis=0)
        print(prediction_1.shape)
        print(prediction_2.shape)
        print(prediction_3.shape)
        return prediction_1, prediction_2, prediction_3
