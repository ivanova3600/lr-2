# Лабораторная работа №1

### Введение
В ходе данной лабораторной работы необходимо реализовать следующие алгоритмы машинного обучения: Linear/ Logistic Regression, SVM, KNN, Naive Bayes в отдельных классах. Также нужно настроить параметры моделей с помощью GridSeachSV или RandomSearchCV, организовать весь процесс с помощью Pipeline. Аналогично проделать с помощью коробочных решений. Также оценить полученные модели.  
### Ход работы
   
#### 1) Линейная регрессия (с градиентным спуском)  
Гиперпараметры - lr (скорость обучения) и epoch (количество эпох).  
Модель: <math>Y = X*w + b</math>  
weight - веса, b - шум, их инициализируем нулями. Гиперпараметры - скорость обучения и количество итераций.   
Во время обучения мы подбираем такие w и b, чтобы модель давала правильные ответы: сначала мы вычисляем y_pred по формуле выше. Далее мы вычисляем функцию потерь:  
  
<img src="https://latex.codecogs.com/svg.image?\frac{1}{2*n}&space;\sum_{i=1}^{n}&space;(y_{pred}-y)^2">   
  
```
np.sum(np.square(y_pred-y))/(2*self.m)
```
self.m - длина входного массива, y - выходные точные данные, y_pred - предсказанные выходные данные. Далее находим градиенты:  
<img src="https://latex.codecogs.com/svg.image?dw=X^T*(y_{y_pred}-y)/n">  
<img src="https://latex.codecogs.com/svg.image?db=&space;\frac{1}{2}&space;\sum_{i=0}^{n}&space;(y_{pred}-y)">    
```
h = np.dot(X, self.w)+self.b
dw = np.dot(X.T,(h-y)) / self.m
db = np.sum(h-y)  / self.m
```
Далее двигаемся в сторону минимального градиента функции:  
```
self.w = self.w - self.lr*dw
self.b = self.b - self.lr*db
```
Предсказание: для этого сначала нормализуем данные:
```
(X-self.mean) / self.std
```
Затем применяем обученную модель:
```
np.dot(X,self.w)+self.b
```
Весь класс LinearRegression_:
```
class LinearRegression_(BaseEstimator, ClassifierMixin):
    def __init__(self, lr = 1, epoch = 5):
        self.lr = lr
        self.epoch = epoch
        self.m = None
                
    def loss(self, y_pred, y):
        return np.sum(np.square(y_pred-y))/(2*self.m)
            
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.m = X.shape[0]   
        self.mean, self.std = X.mean(axis=0), X.std(axis=0)
        
        for i in range(self.epoch):
            
            y_pred = np.dot(X,self.w)
            loss = self.loss(y_pred, y)
            
            h = np.dot(X, self.w)+self.b
            dw = np.dot(X.T,(h-y)) / self.m
            db = np.sum(h-y)  / self.m
            self.w = self.w - self.lr*dw
            self.b = self.b - self.lr*db
        
    def normalizeX(self,X):
        return (X-self.mean) / self.std
        
    def predict(self, X):
        X = self.normalizeX(X)
        return np.dot(X,self.w)+self.b
    
    def transform(self, X):
        return X
```
В качестве оцениваемых параметров выбраны epoch = 1, 2, 5, 7, 10; lr - 0.001, 0.01, 0.1, 0.5, 1. Наиболее высокая точность достигнута при lr = 0.001 и epoch = 1. Использован параметр scoring = 'neg_mean_squared_error', так как у нас задача регрессии:
```
pipe = Pipeline(steps=[('lin', LinearRegression_())])
pipe.get_params().keys()
parameters_grid = {
    'lin__epoch': [1, 2, 5, 7, 10],
    'lin__lr': [0.001, 0.01, 0.1, 0.5, 1],
}

grid_cv = GridSearchCV(pipe, parameters_grid,scoring = 'neg_mean_squared_error')
grid_cv.fit(X_train, y_train)

grid_cv.best_params_
grid_cv.best_score_
```
Вывод:
```
{'lin__epoch': 1, 'lin__lr': 0.001}
-0.048383801291120174
```
Далее оценим точность для лучших параметров. Для этого в отдельной функции metrics вынесены основные метрики: MAE, MSE и RMS:
```
def metrics(y, y_pred):
    print("MSE: ", mean_squared_error(y, y_pred))
    print("MAE: ", mean_absolute_error(y, y_pred))
    print("RMS: ", mean_squared_error(y, y_pred, squared=False))
```
Оценка моей модели:  
```
MSE:  0.049444733426483894
MAE:  0.055220353195027674
RMS:  0.2223617175380778
```
Оценка sklearn.linear_model.LinearRegression:  
```
MSE:  0.043915477586269236
MAE:  0.09326293599310435
RMS:  0.2095602003870707
```
Как я поняла из документации, для регрессии другие указанные метрики (например, confusion matrix) не используются.  Точность у sklearn.linear_model.LinearRegression получилась выше, если рассматривать MSE, если MAE - то построенная модель лучше.
#### 2) Метод опорных векторов  
Модель: <img src="https://latex.codecogs.com/svg.image?class(x)=sign(X*w)">   
  
Нам также необходимо настроить значения w и b. Гиперпараметры - количество итераций и скорость обучения.  
Во время обучения сначала мы строим вектор, определяющий, какой класс у объекта:  
```
y_ = np.where(y > 0, 1, -1)
```
Инициализируем w - нулевой. Далее проходим по всем эпохам, вычисляя condition. Вычисляем значение функции потерь, затем, если ее значение меньше 1, то:  
<img src="https://latex.codecogs.com/svg.image?w=lr*(x[i]*y[i]-2/epoch*w)&space;&plus;&space;w">     
иначе:  
<img src="https://latex.codecogs.com/svg.image?w=w&plus;lr*(-2)/epoch*w">   
```
a = y_[i] * (np.dot(x, self.w))
if a < 1:
    self.w += self.lr * (X[i]*y_[i] - 2/self.epoch*self.w)
else:
    self.w += self.lr * (-2/self.epoch*self.w)
```

Предсказание: применяем к входному массиву модель и далее, если ее значение больше 0, то заменяем на 1, иначе - на 0: 
```
def predict(self, X):
    y_pred = np.dot(X, self.w)
    return np.where(y_pred > 0, 1, 0)
```
Весь класс SWMM:
```
class SVMM(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=1, epoch=1000):
        self.lr = lr        
        self.epoch = epoch

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        y_ = np.where(y > 0, 1, -1)
        
        self.w = np.zeros(n_features)

        for e in range(self.epoch):
            for i, x in enumerate(X):
                a = y_[i] * (np.dot(x, self.w))
                if a < 1:
                    self.w += self.lr * (X[i]*y_[i] - 2/self.epoch*self.w)
                else:
                    self.w += self.lr * (-2/self.epoch*self.w)


    def predict(self, X):
        y_pred = np.dot(X, self.w)
        return np.where(y_pred > 0, 1, 0)
    
    def transform(self, X):
        return X
```
В качестве оцениваемых параметров возьмем lr = 0.001, 0.1, 0.5, 1; epoch = 20, 50, 100, 1000, 5000. Наиболее высокая точность достигнута при epoch = 20 и lr = 0.5:  
```
{'svm__epoch': 20, 'svm__lr': 0.5}
0.7704753121853445
```
Оценка модели: 
```
MSE:  0.05005005005005005
MAE:  0.05005005005005005
RMS:  0.22371868507134143
  
Confusion matrix 
[[949   0]
 [ 50   0]]


Recall score 0.0


roc_auc score 0.5


Accuracy score 0.94994994994995  
```
Оценка sklearn.svm.SVC:  
```
MSE:  0.05005005005005005
MAE:  0.05005005005005005
RMS:  0.22371868507134143  
  
Confusion matrix [[949   0]
 [ 50   0]]


Recall score 0.0


roc_auc score 0.5


Accuracy score 0.94994994994995
```
Таким образом, построенная модель и sklearn.svm.SVC имеют одинаковую точность.  
#### 3) Метод ближайших соседей  
Здесь мы берем k ближайших к рассматриваемому объекту соседей (их значения уже известны) и смотрим, какой из них наиболее часто встречается, и присваиваем этому объекту данный класс.  
Поэтому в обучении в данном методе просто запоминаются тренировочные входная и выходная выборки. Гиперпараметры - число соседей и функция, с помощью которой будем измерять расстояние. У меня их 2: евклидово и абсолютное расстояние.  
<img src="https://latex.codecogs.com/svg.image?euclideandistance=\sqrt{\sum_{i=1}^{n}&space;(v_1&space;-&space;v_2)^2}">     
<img src="https://latex.codecogs.com/svg.image?absolutedistance=\sum_{i=1}^{n}&space;|v_1&space;-&space;v_2|">     
```
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))

def absolute_distance(v1, v2):
    return np.sum(np.absolute(v1-v2))
```
Предсказание: проходим по каждому элементу во входной выборке (X), и вычисляем расстояние до каждой точки:
```
distances = self.distance_func(np.array(self.X[j,:]) , item) 
```
Далее создаем массив длины k - число ближайших соседей и сортируем его в порядке возрастания. Потом находим наиболее частое значение в массиве. Если их несколько, то берем первое.  
'''
dist = np.argsort(point_dist)[:self.k] 
labels = self.y[dist]

lab = mode(labels) 
lab = lab.mode[0]
'''
Полностью весь класс KNN:  
```
class KNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k = 4, distance_func = euclidean_distance):
        self.k = k
        self.distance_func = distance_func
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, x):
        op_labels = []
    
        for item in x: 
            point_dist = []
            for j in range(len(self.X)): 
                distances = self.distance_func(np.array(self.X[j,:]) , item) 
                point_dist.append(distances) 
            
            point_dist = np.array(point_dist) 
            dist = np.argsort(point_dist)[:self.k] 
            labels = self.y[dist]

            lab = mode(labels) 
            lab = lab.mode[0]
            op_labels.append(lab)

        return op_labels
    
    def transform(self, X):
        return X
```
В качестве оцениваемых параметров возьмем k = 2, 5, 7, 10; distance_func = euclidean_distance, absolute_distance. Наиболее высокая точность достигнута при k = 10 и distance_func = euclidean_distance:  
```
{'knn__distance_func': <function __main__.euclidean_distance(v1, v2)>,
 'knn__k': 10}
 0.9509264400048935
```
Полученные оценки моделей:
```
MSE:  0.05205205205205205
MAE:  0.05205205205205205
RMS:  0.22814918814681776

Confusion matrix
 [[947   2]
 [ 50   0]]


Recall score 0.0


roc_auc score 0.49894625922023184


Accuracy score 0.9479479479479479
```
Оценка sklearn.neighbors.KNeighborsClassifier:
```
MSE:  0.05205205205205205
MAE:  0.05205205205205205
RMS:  0.22814918814681776

Confusion matrix
 [[947   2]
 [ 50   0]]


Recall score 0.0


roc_auc score 0.49894625922023184


Accuracy score 0.9479479479479479
```
Таким образом, точности моделей совпадают, но моя модель работает значительно медленнее.  
#### 4) Баессовский классификатор
При реализации данной модели я не использовала гиперпараметры. Здесь вероятность принадлежности к какому-то классу оценивается с помощью апостериорной вероятности:  
<img src="https://latex.codecogs.com/svg.image?y=arg&space;max&space;P(y)&space;*&space;\prod_{i}&space;P(x_i|y)">    
При обучении мы сначала разделяем входные данные на классы (sep_classes): для этого мы проходим по всему набору входных данных, проверяем, если у нас в наборе классов класса с таким именем, то мы его тогда добавляем, иначе просто добавляем в класс текущие входные данные - признаки. То есть мы сформировали классы с признаками:
```
def separate(self, X, y):
    classes = {}
    for i in range(len(X)):
        if y[i] not in classes:
            classes[y[i]] = []
        classes[y[i]].append(X[i])
    return classes
```
Также нам потребуется рассчитать для входных данных среднее отклонение и среднее значение:  
```
def mean_std(self, X):
    for i in zip(*X):
        yield {
           'std' : np.std(i),
            'mean' : np.mean(i)
        }
```
Далее мы проходим по sep_classes, находим для каждого класса априорную вероятность - делим количество элементов в выборке на общее количество элементов, и среднее отклонение, среднее значение:
```
def fit (self, X, y):
    sep_classes = self.separate(X, y)
    self.class_summary = {}
    for i, j in sep_classes.items():
        print(len(j)/len(X))
        self.class_summary[i] = {
            'prior': len(j)/len(X),
            'summ': [k for k in self.mean_std(j)],
        }
```
Для дальнейших действий потребуется распределение Гаусса:  
<img src="https://latex.codecogs.com/svg.image?gausproba&space;=&space;\frac{1}{\sigma&space;\sqrt{2\pi&space;}}&space;\exp&space;{-\frac{(x-\mu&space;)^2}{2\sigma&space;^2}}">  
```
def gaus_distribution(self, x, mean, std):
    return np.exp(-((x-mean)**2 / (2*std**2))) / (np.sqrt(2*np.pi)*std)
```
Предсказание: используем указанную выше формулу апостериорной вероятности. В proba будем записывать вычисленные для каждого класса апостериорную вероятность. Для этого проходим по всем классам, вычисляем сначала гауссовскую плотность (h=1 изначально):
```
for idx in range(ttl_ftrs):
    tmp = row[idx]
    mean = ftrs['summ'][idx]['mean']
    stdev = ftrs['summ'][idx]['std']
    gaus_proba = self.gaus_distribution(tmp, mean, stdev)
    h *= gaus_proba
```
Далее вычисляем апостериорную вероятность принадлежности объекта классу:  
```
proba[cl_name] = ftrs['prior'] * h
```
Потом находим всех вычисленных вероятностей в proba максимум:
```
MAP = max(proba, key = proba.get)
```
Весь метод predict:
```
def predict(self, X):        
    max_apr = []
    
    for row in X:
        proba = {}
            
        for cl_name, ftrs in self.class_summary.items():
            ttl_ftrs =  len(ftrs['summ'])
            h = 1
            for idx in range(ttl_ftrs):
                tmp = row[idx]
                mean = ftrs['summ'][idx]['mean']
                stdev = ftrs['summ'][idx]['std']
                gaus_proba = self.gaus_distribution(tmp, mean, stdev)
                h *= gaus_proba
            proba[cl_name] = ftrs['prior'] * h

        MAP = max(proba, key = proba.get)
        max_apr.append(MAP)

    return max_apr
```
Весь класс Bayes:
```
class Bayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def separate(self, X, y):
        classes = {}
        for i in range(len(X)):
            if y[i] not in classes:
                classes[y[i]] = []
            classes[y[i]].append(X[i])
        return classes

    def mean_std(self, X):
        for i in zip(*X):
            yield {
                'std' : np.std(i), 'mean' : np.mean(i)
            }


    def fit (self, X, y):
        sep_classes = self.separate(X, y)
        self.class_summary = {}

        for i, j in sep_classes.items():
            print(len(j)/len(X))
            self.class_summary[i] = {
                'prior': len(j)/len(X),
                'summ': [k for k in self.mean_std(j)],
            }

    def gaus_distribution(self, x, mean, std):
        return np.exp(-((x-mean)**2 / (2*std**2))) / (np.sqrt(2*np.pi)*std)


    def predict(self, X):        
        max_apr = []

        for row in X:
            proba = {}
            
            for cl_name, ftrs in self.class_summary.items():
                ttl_ftrs =  len(ftrs['summ'])
                h = 1
                for idx in range(ttl_ftrs):
                    tmp = row[idx]
                    mean = ftrs['summ'][idx]['mean']
                    stdev = ftrs['summ'][idx]['std']
                    gaus_proba = self.gaus_distribution(tmp, mean, stdev)
                    h *= gaus_proba
                proba[cl_name] = ftrs['prior'] * h

            MAP = max(proba, key = proba.get)
            max_apr.append(MAP)

        return max_apr
    
    def transform(self, X):
        return X
```
Оценки для Bayes:
```
MSE:  0.12612612612612611
MAE:  0.12612612612612611
RMS:  0.3551424026022887

Confusion matrix
 [[855  94]
 [ 32  18]]


Recall score 0.36


roc_auc score 0.6304741833508956


Accuracy score 0.8738738738738738
```
Оценки для sklearn.naive_bayes.GaussianNB:
```
MSE:  0.12612612612612611
MAE:  0.12612612612612611
RMS:  0.3551424026022887
Confusion matrix
 [[855  94]
 [ 32  18]]


Recall score 0.36


roc_auc score 0.6304741833508956


Accuracy score 0.8738738738738738
```
Таким образом, в данном случае обе модели получились равнозначны.
  
### Вывод
Таким образом, с учетом скорости работы, для данной задачи лучше всего использовать или линейную регрессию, или метод опорных векторов (с методом ближайших соседей точность примерно одинакова, но SVM работает значительно быстрее).
