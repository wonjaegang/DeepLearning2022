import numpy as np
import csv


class AbaloneSLP:
    def __init__(self, epoch_count, mb_size, report_period):
        np.random.seed(1234)

        self.epoch_count = epoch_count
        self.mb_count = None
        self.mb_size = mb_size
        self.report_period = report_period

        self.random_mean = 0
        self.random_std = 0.0030
        self.train_per_data = 0.8

        self.learning_rate = 0.001

        self.input_cnt = 10
        self.output_cnt = 1

        self.shuffle_map = None
        self.test_start_index = None

        self.data = None
        self.weight = None
        self.bias = None

    def exec(self):
        self.load_data()
        self.init_model()
        self.train_and_test()

    def load_data(self):
        with open('Abalone_Data.csv') as csvFile:
            csvReader = csv.reader(csvFile)
            next(csvReader, None)
            rows = []
            for row in csvReader:
                rows.append(row)

        self.data = np.zeros([len(rows), self.input_cnt + self.output_cnt])

        for n, row in enumerate(rows):
            if row[0] == 'I':
                self.data[n, 0] = 1
            if row[0] == 'M':
                self.data[n, 1] = 1
            if row[0] == 'F':
                self.data[n, 2] = 1
            self.data[n, 3:] = row[1:]

    def init_model(self):
        self.weight = np.random.normal(self.random_mean, self.random_std, [self.input_cnt, self.output_cnt])
        self.bias = np.zeros([self.output_cnt])

    def train_and_test(self):
        self.arrange_data()
        test_x, test_y = self.get_test_data()

        for epoch in range(self.epoch_count):
            losses, accuracies = [], []

            for mb_index in range(self.mb_count):
                train_x, train_y = self.get_train_data(mb_index)
                loss, accuracy = self.run_train(train_x, train_y)
                losses.append(loss)
                accuracies.append(accuracy)

            if self.report_period and (epoch + 1) % self.report_period == 0:
                test_accuracy = self.run_test(test_x, test_y)
                print("Epoch %d: Training mean loss: %.6f, Training mean accuracy: %.6f, Test accuracy: %f"
                      % (epoch + 1, np.mean(losses), np.mean(accuracies), test_accuracy))

        final_accuracy = self.run_test(test_x, test_y)
        print("\nFinal Test accuracy: %f" % final_accuracy)

    def arrange_data(self):
        self.mb_count = int(len(self.data) * self.train_per_data) // self.mb_size
        self.test_start_index = self.mb_count * self.mb_size

        self.shuffle_map = np.arange(self.data.shape[0])
        np.random.shuffle(self.shuffle_map)

    def get_test_data(self):
        test_data = self.data[self.shuffle_map[self.test_start_index:]]
        return test_data[:, : -self.output_cnt], test_data[:, -self.output_cnt:]

    def get_train_data(self, mb_index):
        if mb_index == 0:
            np.random.shuffle(self.shuffle_map[: self.test_start_index])
        train_data = self.data[self.shuffle_map[self.mb_size * mb_index: self.mb_size * (mb_index + 1)]]
        return train_data[:, : -self.output_cnt], train_data[:, -self.output_cnt:]

    def run_train(self, x, y):
        output, aux_data_nn = self.forward_neuralnet(x)
        loss, aux_data_pp = self.forward_postproc(output, y)

        accuracy = self.eval_accuracy(output, y)

        dL_dY = self.backprop_postproc(aux_data_pp)
        self.backprop_neuralnet(dL_dY, aux_data_nn)

        return loss, accuracy

    def run_test(self, x, y):
        output, _ = self.forward_neuralnet(x)
        return self.eval_accuracy(output, y)

    def forward_neuralnet(self, X):
        output = np.matmul(X, self.weight) + self.bias
        return output, X

    def forward_postproc(self, output, Y):
        diff = output - Y
        loss = np.mean(np.square(diff))
        return loss, diff

    def backprop_postproc(self, diff):
        dL_dY = 2 * diff / (self.mb_size * self.output_cnt)
        return dL_dY

    def backprop_neuralnet(self, dL_dY, X):
        dY_dW = X.transpose()
        dY_db = np.ones([self.mb_size])

        self.weight -= self.learning_rate * np.matmul(dY_dW, dL_dY)
        self.bias -= self.learning_rate * np.matmul(dY_db, dL_dY)

    def eval_accuracy(self, output, Y):
        return 1 - np.mean(np.abs((output - Y) / Y))


if __name__ == '__main__':
    a = AbaloneSLP(epoch_count=10, mb_size=10, report_period=1)
    a.exec()
