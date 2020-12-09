import face_recognition
import os
import cv2
import pickle
import ams.variables as var


def get_encodings(path):
    input_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return face_recognition.face_encodings(input_img)


class extract:
    def __init__(self, input_path, output_pickle_path):
        self.input_dir = input_path
        self.labels = []
        self.encodings = []
        self.pickle_file = output_pickle_path

    def extract(self):
        if not os.path.exists(self.input_dir):
            print("data-set not found!", self.input_dir)
            return
        print("[Info] Extracting...")
        labels = os.listdir(self.input_dir)
        for i in range(len(labels)):
            label = labels[i]
            images = os.listdir(self.input_dir + "/" + label)
            for img in images:
                img_path = self.input_dir + "/" + label + "/" + img
                self.labels.append(label)
                self.encodings.append(get_encodings(img_path))
            print(f'{round(((i + 1) / len(labels)) * 100, 2)}% completed')
        print("[Info] Extraction completed")
        self.save_encodings()

    def save_encodings(self):
        print("[INFO] saving encodings...")
        data = {"labels": self.labels, "encodings": self.encodings}
        f = open(self.pickle_file, "wb")
        f.write(pickle.dumps(data))
        f.close()


if __name__ == "__main__":
    train_encodings = extract(var.train_data_set_path_wrt_sub_folder, var.train_encodings_model_path_wrt_sub_folder)
    train_encodings.extract()
    verify_encodings = extract(var.verify_data_set_path_wrt_sub_folder, var.verify_encodings_model_path_wrt_sub_folder)
    verify_encodings.extract()
