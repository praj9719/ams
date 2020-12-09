import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import ams.variables as var

failed_to_load_x_index = []


class train:
    def __init__(self):
        self.data = pickle.loads(open(var.train_encodings_model_path_wrt_sub_folder, "rb").read())
        self.x = self.__filter_x__()
        self.y = self.data["labels"]
        self.__handle_none__()
        self.classes = list(set(self.y))
        print("Classes:", len(self.classes))

    def __filter_x__(self):
        x_new = []
        for list_ele in self.data["encodings"]:
            if len(list_ele) is 0:
                x_new.append(None)
            for row in list_ele:
                x_new.append(row)
        return x_new

    def __filter_y__(self):
        new_y = []
        for val in self.data["labels"]:
            temp_row = [val]
            new_y.append(temp_row)
        return new_y

    def __handle_none__(self):
        x = self.x
        y = self.y
        if len(x) != len(y):
            print(f'Error!!, Invalid data-set len_x: {len(x)} len_y: {len(y)}')
            return
        for i in range(len(x) - 1, -1, -1):
            if x[i] is None:
                x.pop(i)
                y.pop(i)
        self.x = x
        self.y = y

    def knn(self):
        print("Training with KNN init")
        x = self.x
        y = self.y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)
        print("K = " + str(len(self.classes)))
        knn = KNeighborsClassifier(n_neighbors=len(self.classes))
        print("Knn initialised")
        knn.fit(x_train, y_train)
        print("Model trained")
        predict = knn.predict(x_test)
        print(classification_report(y_test, predict))
        print("Accuracy: ", accuracy_score(y_test, predict))
        print("Saving model")
        with open(var.trained_knn_model_path_wrt_sub_folder, 'wb') as file:
            pickle.dump(knn, file)
        print("Model saved!")

    def svm(self):
        print("Training with SVM init")
        x = self.x
        y = self.y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)
        print("Training...")
        linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(x_train, y_train)
        rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(x_train, y_train)
        poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(x_train, y_train)
        sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(x_train, y_train)
        print("Trained")
        accuracy_lin = linear.score(x_test, y_test)
        accuracy_poly = poly.score(x_test, y_test)
        accuracy_rbf = rbf.score(x_test, y_test)
        accuracy_sig = sig.score(x_test, y_test)
        print("Accuracy Linear Kernel:", accuracy_lin)
        print("Accuracy Polynomial Kernel:", accuracy_poly)
        print("Accuracy Radial Basis Kernel:", accuracy_rbf)
        print("Accuracy Sigmoid Kernel:", accuracy_sig)
        print("Saving model")
        with open(var.trained_svm_linear_model_path_wrt_sub_folder, 'wb') as file:
            pickle.dump(linear, file)
        with open(var.trained_svm_poly_model_path_wrt_sub_folder, 'wb') as file:
            pickle.dump(poly, file)
        with open(var.trained_svm_rbf_model_path_wrt_sub_folder, 'wb') as file:
            pickle.dump(rbf, file)
        with open(var.trained_svm_sig_model_path_wrt_sub_folder, 'wb') as file:
            pickle.dump(sig, file)
        print("Model Saved")


if __name__ == "__main__":
    train = train()
    train.knn()
    train.svm()
