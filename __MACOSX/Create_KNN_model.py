import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
from Cale_K import cale

if __name__ == "__main__":
    _ = 'xiaoxinxin.csv'
    test_data = pd.read_csv(_, header=0)
    x, y = test_data.iloc[:, 2:], test_data.iloc[:, 1]
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)  # 25%数据用于训练

    knn = KNeighborsClassifier(cale())
    knn.fit(x_train, y_train)
    # 然后先执行fit方法进行拟合操作得出模型,将训练数据集作为参数传入
    # 返回一个History的对象，记录了loss和其他指标的数值随epoch变化的情况

    joblib.dump(knn, 'model_knn.pkl')
    # print(knn.score(x_test, y_test))
