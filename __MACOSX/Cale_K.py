import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def pr(k_range, _):
    plt.plot(k_range, _)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


def cale():
    m = '../__MACOSX/xiaoxinxin.csv'
    test_data = pd.read_csv(m, header=0)
    x, y, _ = test_data.iloc[:, 2:], test_data.iloc[:, 1], []  # 数据, 标签, 对应k的得分
    k_range = [i for i in range(1, 32)]
    print(k_range)
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        grade = cross_val_score(knn, x, y, cv=5, scoring='accuracy')
        _.append(grade.mean())
    # pr(k_range, _)
    # print(_)
    return _.index(max(_)) + 1


if __name__ == "__main__":
    print(cale())
