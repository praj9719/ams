import tkinter
import cv2
import math
import csv
import time
import threading
import datetime
import random
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
from tkinter.ttk import Style
from ams.main.predict import predict
import ams.variables as var

holder = var.image_holder_path
model_path = var.trained_svm_linear_model_path
verify_path = var.verify_encodings_model_path
csv_path = var.csv_output_path
faceCascade = cv2.CascadeClassifier(var.haar_cascade_model_path)


def box_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame


class ams(object):
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.title('Attendance management system')
        self.root.geometry("1170x530")
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.resizable(width=False, height=False)

        self.model = predict(model_path, verify_path)

        self.list_name = []

        self.img_canvas = tkinter.Label(self.root, relief="solid")
        self.img_canvas.place(x=25, y=25)
        self.text_label = tkinter.Label(self.root, text="Attendance Management System",
                                        font="Helvetica 16 bold italic")
        self.text_label.place(x=750, y=30)
        self.attendance_box = tkinter.Listbox(self.root, bg="#f0f0f0", selectbackground="#f0f0f0", bd=0,
                                              selectforeground="#000", font="Consolas 14",
                                              highlightbackground="#f0f0f0")
        self.attendance_box.place(x=700, y=100, height="250", width="440")
        self.btn_cam = tkinter.Button(self.root, text="Camera", width="15", bg="#5b32f0",
                                      fg="#fff", font="bold", command=lambda: self.init_video(0))
        self.btn_cam.place(x=725, y=390)
        self.btn_import = tkinter.Button(self.root, text="Import", width="15", bg="#5b32f0",
                                         fg="#fff", font="bold", command=self.import_media)
        self.btn_import.place(x=945, y=390)
        self.btn_update_csv = tkinter.Button(self.root, text="Update CSV", width="35", bg="#5b32f0",
                                             fg="#fff", font="bold", command=self.update_csv)
        self.btn_update_csv.place(x=725, y=460)

        self.pre_delay = 40

        self.cap = cv2.VideoCapture()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
        self.__release_video__()

    def import_media(self):
        if self.cap.isOpened():
            self.__release_video__()
        path = filedialog.askopenfilename(title='open')
        if path is not "":
            extension = path[-4:]
            if extension == ".jpg":
                self.update_canvas(Image.open(path))
                threading.Thread(target=self.predict, args=(cv2.imread(path), True)).start()
            elif extension == ".mp4":
                self.init_video(path)
            else:
                messagebox.showwarning("Unable to open this file", f'{extension} format not supported!')

    def init_video(self, path):
        if self.cap.isOpened():
            self.__release_video__()
            return
        self.cap = cv2.VideoCapture(path)
        threading.Thread(target=self.capture_frame).start()

    def capture_frame(self):
        if self.cap.isOpened():
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            if frame is not None:
                self.pre_delay = self.pre_delay + 1
                if self.pre_delay == 50:
                    threading.Thread(target=self.predict, args=(frame, False)).start()
                    self.pre_delay = 0
                frame = box_faces(frame)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                self.update_canvas(img)
                self.img_canvas.after(10, self.capture_frame)

    def update_canvas(self, image):
        img = image.resize((620, 470), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(image=img)
        self.img_canvas.img_tk = img_tk
        self.img_canvas.configure(image=img_tk)

    def __release_video__(self):
        self.cap.release()
        self.update_canvas(Image.open(holder))

    def predict(self, image, b):
        self.model.extract(image)
        encodings = self.model.encodings
        if len(encodings) == 0:
            print("Face not fount")
        else:
            result = self.model.predict()
            for r in result:
                self.update_table(r)
            if b:
                self.plot_landmarks(image)

    def plot_landmarks(self, image):
        landmarks = self.model.landmarks
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        thickness = 1
        if image.shape[0] > 800 or image.shape[1] > 800:
            thickness = 5
        elif image.shape[0] > 400 or image.shape[1] > 400:
            thickness = 2
        for i in range(len(landmarks)):
            for mark in landmarks[i]:
                points = landmarks[i][mark]
                last_pt = None
                for point in points:
                    cv2.circle(image, (point[0], point[1]), radius=0, color=(100, 255, 100), thickness=thickness)
                    if last_pt is not None:
                        cv2.line(image, point, last_pt, color=(100, 255, 100))
                        pass
                    last_pt = point
        img = Image.fromarray(image, 'RGB')
        self.update_canvas(img)

    def update_table(self, result):
        if result[0]:
            name = result[1]
        else:
            name = f'Unknown ({result[1]})'
        t = datetime.datetime.now()
        c_time = f'{t.hour}:{t.minute}:{t.second}'  # 2020/12/02 10:40:12
        space = " "
        limit = 30
        f_name = f'{name}{space * (limit - len(name))}'
        f_time = f'{c_time}{space * (12 - len(c_time))}'
        self.attendance_box.insert(0, f' {f_time}{f_name}  ')
        if result[0]:
            self.update_list(name)

    def update_list(self, name):
        if name in self.list_name:
            return
        self.list_name.append(name)

    def update_csv(self):
        names = self.model.classes
        names = [name.lower() for name in names]
        attendance = {"ams": ["ams"]}
        for name in names:
            attendance[name] = [name]
        t = datetime.datetime.now()
        today = str(f"{t.day}-{t.month}")
        try:
            open(csv_path, 'a')
            with open(csv_path, 'r') as read_file:
                reader = csv.reader(read_file)
                for row in reader:
                    try:
                        attendance[row[0]] = row
                    except Exception as e:
                        print("Error", e)
                read_file.close()
                with open(csv_path, 'w') as write_file:
                    writer = csv.writer(write_file, lineterminator='\n')
                    if attendance["ams"][-1] != today:
                        for e in attendance:
                            attendance[e].append("-")
                    position = len(attendance["ams"])-1
                    attendance["ams"][position] = today
                    for name in self.list_name:
                        name = name.lower()
                        attendance[name][position] = "1"
                    writer.writerow(attendance["ams"])
                    names.sort()
                    for name in names:
                        writer.writerow(attendance[name])
                    write_file.close()
                    messagebox.showinfo("File Saved!", f'Attendance sheet is updated at {csv_path}')
        except Exception as e:
            print("Error", e)
            messagebox.showwarning("Close csv sheet if already opened!", f'{e}')

    def execute(self):
        self.root.mainloop()
