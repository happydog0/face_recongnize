from tkinter import *
import pandas as pd
from PIL import Image, ImageTk
import cv2
import joblib
from Play_Gif import playGif
import dlib
import tkinter.filedialog


class Apk:
    def __init__(self, _):
        self.root = _
        self.width = 700
        self.height = 400
        self.root.geometry("700x400+500+200")  # 窗口大小 (宽度x高度)+屏幕(x轴+y轴)
        self.root.title('情绪识别系统')  # 窗口标题
        # 首页按键
        self.label = Label(self.root, text="人脸情绪识别系统", font=("华文行楷", 50), fg="red")
        self.button = Button(self.root, text='拍照识别', font=("方正黑体简体", 30), command=self.open_camera, bg="red")
        self.button1 = Button(self.root, text='开始检测', font=("方正黑体简体", 30), command=self.recognize, bg="blue")
        self.button2 = Button(self.root, text='退出系统', font=("方正黑体简体", 30), command=self.delete, bg="green")
        # 拍照界面按键
        self.camera_frame = Label(self.root)
        self.open_identify = Button(self.root, text="实时识别", font=("方正黑体简体", 20), command=self.open_identify)
        self.close_identify = Button(self.root, text="关闭识别", font=("方正黑体简体", 20), command=self.close_identify)
        self.capture_button = Button(self.root, text="拍照", font=("方正黑体简体", 20), command=self.capture)
        self.exit_camera_button = Button(self.root, text="退出拍照", font=("方正黑体简体", 20), command=self.close_camera)
        # 识别界面按键
        self.recognize_results = Label(self.root)
        self.select_road = Button(self.root, text="选择路径", font=("方正黑体简体", 20), command=self.select_road, bg="green")
        self.exit_recognize = Button(self.root, text='返回', font=("方正黑体简体", 20), command=self.recognize_exit, bg="green")
        # 首页gif
        self.giff = Button(self.root)
        self.root.protocol('WM_DELETE_WINDOW', self.delete)  # 【重要】关闭窗口后的事件：delete
        self.gif = playGif("drink.gif")
        self.gif.playGif(self.root, self.giff)  # 实现动态播放
        # 初始化
        self.predictor = dlib.shape_predictor('../__MACOSX/shape_predictor_68_face_landmarks.dat')
        self.detector = dlib.get_frontal_face_detector()
        self.model = joblib.load(filename='model_knn.pkl')
        self.mark = ['angry', ' disgust', 'surprise', 'fear', 'neutral', 'happy', 'sad']
        self.path = "captured_face.jpg"
        self.camera_open = False  # 开启摄像头标记
        self.real_time = False  # 实时识别标记
        self.sum = None
        self.cap = None  # 相机
        self.photo = None  # tk图片
        self.frame = None  # bgr图像或者PIL像素
        self.open1()  # 打开首页和gif
        self.root.mainloop()

    def delete(self):  # 删除gif的临时图
        self.root.destroy()
        self.gif.close()

    def re_resize(self):  # 对pil图片缩放
        scale_width = self.width / self.frame.width
        scale_height = self.height / self.frame.height
        scale = min(scale_width, scale_height)
        self.frame = self.frame.resize((int(self.frame.width * scale), int(self.frame.height * scale)))
        self.photo = ImageTk.PhotoImage(self.frame)

    def open1(self):  # 打开首页和gif
        self.label.pack()
        self.giff.pack(side=LEFT)
        self.button.pack(padx=10, pady=10)
        self.button1.pack(padx=10, pady=10)
        self.button2.pack(padx=10, pady=10)

    def close1(self):  # 隐藏首页和gif
        self.label.pack_forget()
        self.button.pack_forget()
        self.button1.pack_forget()
        self.button2.pack_forget()
        self.giff.pack_forget()

    def open2(self):  # 打开拍照界面
        self.camera_frame.pack(side=LEFT)
        self.open_identify.pack(padx=10, pady=10)
        self.close_identify.pack(padx=10, pady=10)
        self.capture_button.pack(padx=10, pady=10)
        self.exit_camera_button.pack(padx=10, pady=10)

    def close2(self):  # 关闭拍照界面
        self.open_identify.pack_forget()
        self.close_identify.pack_forget()
        self.camera_frame.pack_forget()
        self.capture_button.pack_forget()
        self.exit_camera_button.pack_forget()

    def open3(self):  # 打开识别界面
        self.recognize_results.pack(side=LEFT)
        self.select_road.pack(padx=10, pady=10)
        self.exit_recognize.pack(padx=10, pady=10)

    def close3(self):  # 关闭识别界面
        self.exit_recognize.pack_forget()
        self.select_road.pack_forget()
        self.recognize_results.pack_forget()

    def open_camera(self):  # 打开相机
        self.close1()  # 关闭首页
        self.cap = cv2.VideoCapture(0)  # 开镜
        self.camera_open = True  # 标记
        self.update_camera()  # 标记之后开始更新
        self.open2()  # 打开拍照组件

    def close_camera(self):  # 关相机
        self.close2()  # 关闭拍照组件
        self.cap.release()  # 释放镜头
        self.camera_open = False  # 标记
        self.open1()  # 打开首页

    def update_camera(self):  # 更新画面
        if self.camera_open:
            ret, self.frame = self.cap.read()  # frame是像素值，多维数组
            self.frame = cv2.flip(self.frame, 1)
            if self.real_time: self.face_identify()
            self.frame = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))  # bgr->rgb->pil
            self.re_resize()
            self.photo = ImageTk.PhotoImage(self.frame)  # 转化为tk显示的对象
            self.camera_frame.config(image=self.photo)
            root.after(5, self.update_camera)  # 时间更新

    def capture(self):  # 拍照
        if self.camera_open:
            ret, self.frame = self.cap.read()
            cv2.imwrite("captured_face.jpg", self.frame)
            self.close_camera()

    def open_identify(self):  # 打开实时识别
        self.real_time = True

    def close_identify(self):  # 关闭实时识别
        self.real_time = False

    def recognize(self):
        self.close1()  # 关首页
        self.frame = cv2.imread(self.path)  # numpy的三维数组BGR，(blue,green,red)
        self.face_identify()
        self.frame = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))  # 先转化RGB,在转化为PIL对象
        self.re_resize()
        self.recognize_results.config(image=self.photo)
        self.open3()  # 开识别结果页面

    def recognize_exit(self):  # 识别退出
        self.open1()  # 打开首页
        self.close3()  # 关闭识别结果页面

    def select_road(self):
        self.path = tkinter.filedialog.askopenfilename().replace("/", "\\\\")
        if self.path:
            self.close3()
            self.recognize()
        else:
            self.path = 'captured_face.jpg'

    def face_identify(self):  # 对单张图片识别
        def cale1(a, b):
            return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

        def cale2(a, y):
            return ((a.x - y) ** 2 + (a.y - y) ** 2) ** 0.5

        faces = self.detector(self.frame, 0)
        for __, _ in enumerate(faces):
            shape = self.predictor(self.frame, _)
            for index, pt in enumerate(shape.parts()):
                cv2.circle(self.frame, (pt.x, pt.y), 1, (0, 255, 0), -1)
            cv2.rectangle(self.frame, (_.left(), _.top()), (_.right(), _.bottom()), (0, 0, 225), 1)  # 画矩阵框
            face_width = _.right() - _.left()
            # 嘴巴裂开程度
            mouth_size = cale1(shape.part(49), shape.part(55))
            features1 = round(mouth_size / face_width, 2)
            # 嘴巴张开程度
            mouth_open = 0
            for i in range(50, 55):
                mouth_open += cale1(shape.part(i), shape.part(110 - i))
            features2 = round(mouth_open / 5 / face_width, 2)
            # 挑眉程度
            eyebrow_raising = 0
            for i in range(18, 28):
                eyebrow_raising += cale2(shape.part(i), _.top())
            features3 = round(eyebrow_raising / 10 / face_width, 2)
            # 皱眉程度
            frown = 0
            for i in range(18, 23):
                frown += cale1(shape.part(i), shape.part(i + 5))
            features4 = round(frown / 5 / face_width, 2)
            # 眼睛张开程度
            eyes_open = 0
            ___ = [(38, 42), (39, 41), (44, 48), (45, 47)]
            for i, j in ___:
                eyes_open += cale1(shape.part(i), shape.part(j))
            features5 = round(eyes_open / 4 / face_width, 2)
            ans = self.model.predict(pd.DataFrame([[features1, features2, features3, features4, features5]]))
            cv2.putText(self.frame, f"face_{__}:{self.mark[ans[0]]}", (_.left(), _.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 225), 1, cv2.LINE_AA)


if __name__ == "__main__":
    # 数据打包问题，打包的时候有特征标签，这里没有导致的报错，可以不用管
    root = Tk()
    Apk(root)
