import face_recognition
import pickle
import cv2


class predict(object):
    def __init__(self, model_path, verify_path):
        self.model_path = model_path
        self.face_location = []
        self.landmarks = []
        self.encodings = []
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)

        self.data = pickle.loads(open(verify_path, "rb").read())
        self.x = self.__filter_x__()
        self.y = self.data["labels"]
        self.__handle_none__()
        self.classes = list(set(self.y))

    def __filter_x__(self):
        x_new = []
        for list_ele in self.data["encodings"]:
            if len(list_ele) is 0:
                x_new.append(None)
            for row in list_ele:
                x_new.append(row)
        return x_new

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

    def predict(self):
        predictions = self.model.predict(self.encodings)
        return self.verify(predictions)

    def extract(self, image):
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.face_location = face_recognition.face_locations(input_image)
        self.landmarks = face_recognition.face_landmarks(input_image, self.face_location)
        self.encodings = face_recognition.face_encodings(input_image, self.face_location)

    def verify(self, predictions):
        result = []
        if len(self.encodings) != len(predictions):
            print(f"number of predictions: {len(predictions)} is not equal to number of encodings: {len(self.encodings)}")
            return result
        for i in range(len(predictions)):
            prediction = predictions[i]
            encoding = self.encodings[i]
            comp_encodings = []
            for j in range(len(self.y)):
                if self.y[j] == prediction:
                    comp_encodings.append(self.x[j])
            matches = []
            for j in range(len(comp_encodings)):
                matches.append(face_recognition.compare_faces(encoding, [comp_encodings[j]])[0])
            b = False
            if sum(matches) > len(matches)//2:
                b = True
            result.append([b, prediction])
        return result


