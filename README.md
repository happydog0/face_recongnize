# face_recongnize

成都东软学院人工智能系大二小学期基于`opencv`，`dlib`的人脸情绪识别的小项目，仅供参考学习

* `data`文件夹为训练数据集，解压图片到当前路径即可，数据集为The Japanese Female Facial Expression (JAFFE) Database
* `__MACOSX`为主文件目录
* `Data_calassification.py`是模型数据的预处理，按照`2：8`打包
* `Cale_K.py`是计算训练数据集的`K`值
* `Play_Gif.py`是`tk`界面中放映`gif`的程序
* `Creat_KNN_model.py`是创建`knn`模型
* `Create_root.py`是主程序
* `xiaoxinxin.csv`为数据预处理的表格
* `shape_predictor_68_face_landmarks.dat`为人脸的`68`个点位的识别器
* `model_knn.pkl`为打包之后的模型
* `dlib-19.17.99-cp37-cp37m-win_amd64.whl`为`python==3.7`版本适应的`dlib` 
* `dlib-19.19.0-cp38-cp38-win_amd64.whl.whl`为`python==3.8`版本适应的`dlib` 

## 环境要求

$python\leq3.8$ 

建议使用`miniconda` 

* `pip`更新
  * `pip install --upgrade pip`

* 安装环境
  * `conda create -n py37 python==3.7 -y`

* 安装`dlib`，需要`cd`到`.whl`文件位置之后

  * `python==3.8`运行
    * `pip install dlib-19.19.0-cp38-cp38-win_amd64.whl.whl `

  * `python==3.7`运行
    * `pip install dlib-19.17.99-cp37-cp37m-win_amd64.whl `

* 安装`opencv`
  * `pip install opencv-python`
* 安装`PIL`
  * `pip install pillow`
  * `conda install pillow`
* 安装`joblib`
  * `pip install joblib`
* 安装`matplotlib`
  * `pip install matplotlib`
  * `conda install matplotlib`
* 安装`sklearn`
  * `pip install  scikit-learn`
  * 在`python==3.8`版本中需要降级安装`sklearn`
* 安装`pandas`
  * `pip install pandas`

`双击运行.exe`要求：

* `conda`环境名为`py37` 
* 有一个`captured_face.jpg`的初始化图片

$eg:参考学习$ 
