import os
import cv2
import dlib
import pandas as pd


def rev(_):
    __ = ['AN', 'DI', 'SU', 'FE', 'NE', 'HA', 'SA']
    return __.index(_)


def cale1(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def cale2(a, y):
    return ((a.x - y) ** 2 + (a.y - y) ** 2) ** 0.5


def cale(_):  # 传入图片路径

    # 数据预处理务必注释掉下面两行代码
    detector = dlib.get_frontal_face_detector()  # 人脸分类器
    predictor = dlib.shape_predictor('../__MACOSX/shape_predictor_68_face_landmarks.dat')  # 获取68点位人脸检测器

    img = cv2.imread(_)
    # img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)  #### 如果原图为彩色，需要转化为灰度
    faces = detector(img, 0)  # 对图片画人脸框，返回矩形框4个点坐标，传入路径和放大倍数，越大用于小脸检测;返回脸部检测为矩阵框数值
    # 坐标为[(x1, y1)(x2, y2)]。通过函数的left, right, top, bottom方法分别获取对应的x1, x2, y1, y2值
    ls, cnt = [[]], 0
    for __, _ in enumerate(faces):  # 返回矩阵索引和矩阵值
        cnt += 1
        ls.append([])

        # 画出68点位
        shape = predictor(img, _)  # 输入灰度图片和预测的边界框，返回68个关键点位置的迭代器
        for index, pt in enumerate(shape.parts()):  # 迭代器:关键点标号，关键点坐标
            cv2.circle(img, (pt.x, pt.y), 1, (0, 255, 0), -1)  # 给定点画圆，图片路径，圆心坐标，半径，颜色，圆轮廓
        cv2.rectangle(img, (_.left(), _.top()), (_.right(), _.bottom()), (0, 0, 225), 1)  # 画矩阵框
        cv2.putText(img, f"face_{cnt - 1}", (_.left(), _.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 1,
                    cv2.LINE_AA)
        cv2.imwrite('Identify_results.jpg', img)

        face_width = _.right() - _.left()  # 图片中第__张脸的宽度

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

        ___ = [features1, features2, features3, features4, features5]
        for i in ___: ls[__].append(i)
    ls.pop()
    # 显示图片取消下面两行代码注释
    # cv2.imshow('test', img)
    # cv2.waitKey(2)
    return ls, cnt  # 返回特征计算和人脸数


def batch_process():
    path_imgs = '../data'  # 图片文件夹路径
    cnt = 0
    for pn in os.listdir(path_imgs):
        if pn.endswith(".tiff"):
            ans.append([])
            cnt += 1
            ans[cnt].append(pn)  # 添加名字
            ans[cnt].append(rev(pn.split('.')[1][:-1]))  # 添加标签
            __, tot = cale(path_imgs + "/" + pn)
            for _ in __[0]: ans[cnt].append(_)  # 添加一张人脸计算数据
    ans.pop()
    pd.DataFrame(ans).to_csv('xiaoxinxin.csv', index=False, header=False)


if __name__ == "__main__":

    detector = dlib.get_frontal_face_detector()  # 人脸分类器
    predictor = dlib.shape_predictor('../__MACOSX/shape_predictor_68_face_landmarks.dat')  # 获取68点位人脸检测器
    __ = ['name', 'mark', 'value1', 'value2', 'value3', 'value4', 'value5']  # 表格键值预处理
    ans = [[]]  # 输出表格
    for _ in __: ans[0].append(_)  # 表格预处理
    batch_process()
