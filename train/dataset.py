import os
import cv2
import random
import ams.variables as var


def validate_dir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
            return True
        except Exception as e:
            print(e)
            return False
    return True


class data:
    def __init__(self):
        self.input_dir = var.initial_input_data_set_path
        self.output_dir = var.data_set_path_wrt_sub_folder
        self.output_train = var.train_data_set_path_wrt_sub_folder
        self.output_test = var.test_data_set_path_wrt_sub_folder
        self.output_verify = var.verify_data_set_path_wrt_sub_folder

    def obtain(self, max_classes=None, max_train_data=None, max_test_data=1, max_verify_data=3):
        if not self.validate():
            print("Error!! Failed to load directories!")
            return
        folders = os.listdir(self.input_dir)
        print("Available classes: " + str(folders))
        if max_classes is None or max_classes > len(folders):
            max_classes = len(folders)
        for i in range(max_classes):
            images = os.listdir(self.input_dir + "/" + folders[i])
            # Train
            if max_train_data is None or max_train_data > len(images):
                max_train_data = len(images)
            for j in range(max_train_data):
                self.save(folders[i], images[j], self.output_train + "/" + folders[i][5:], resize=True)
            # Test
            if max_test_data is None or max_test_data > len(images):
                max_test_data = 1
            for j in range(max_test_data):
                r = random.randint(0, len(images) - 1)
                self.save(folders[i], images[r], self.output_test, resize=False)
            # Verify
            if max_verify_data is None or max_verify_data > len(images):
                max_verify_data = 3
            for j in range(max_verify_data):
                r = random.randint(0, len(images) - 1)
                self.save(folders[i], images[r], self.output_verify + "/" + folders[i][5:], resize=True)
            print(f'{round(((i + 1) / max_classes) * 100, 2)}% completed')

    def save(self, folder_name, image_name, output_path, resize):
        img = cv2.imread(self.input_dir + "/" + folder_name + "/" + image_name)
        if resize:
            img = cv2.resize(img, (100, 100))
        validate_dir(output_path)
        img_path = output_path + "/" + image_name
        cv2.imwrite(img_path, img)

    def validate(self):
        return validate_dir(self.input_dir) and validate_dir(self.output_dir) and \
               validate_dir(self.output_train) and validate_dir(self.output_test) and \
               validate_dir(self.output_verify) and validate_dir(var.models_path_wrt_sub_folder)


if __name__ == "__main__":
    data = data()
    data.obtain(max_train_data=20, max_test_data=2)
