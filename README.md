# Лабораторная работа №1

<img src="https://latex.codecogs.com/svg.latex?x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}"/>


### Введение
В ходе данной лабораторной работы необходимо реализовать следующие алгоритмы машинного обучения: Linear/ Logistic Regression, SVM, KNN, Naive Bayes в отдельных классах. Также нужно настроить параметры моделей с помощью GridSeachSV или RandomSearchCV, организовать весь процесс с помощью Pipeline. Аналогично проделать с помощью коробочных решений. Также оценить полученные модели.  
### Ход работы
1) Линейная регрессия (с градиентным спуском)  
Гиперпараметры - lr (скорость обучения) и epoch (количество эпох).  
Модель: <math>Y = X*w + b</math>  
weight - веса, b - шум, их инициализируем нулями. Гиперпараметры - скорость обучения и количество итераций.   
Во время обучения мы подбираем такие w и b, чтобы модель давала правильные ответы: сначала мы вычисляем y_pred по формуле выше. Далее мы вычисляем функцию потерь:  
<img src="https://latex.codecogs.com/svg.latex?loss=\fraq{1}{2*n}\sum_{i=1}^n(y_{pred}-y)"/>   
```
np.sum(np.square(y_pred-y))/(2*self.m)
```
self.m - длина входного массива, y - выходные точные данные, y_pred - предсказанные выходные данные. Далее находим градиенты:  
<img src="https://latex.codecogs.com/svg.latex?dw=X^T*(y_{y_pred}-y)/n"/>  
<img src="https://latex.codecogs.com/svg.latex?db=1/2*\sum_{i=0}^{n} (y_{pred}-y)"/>  
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
В качестве оцениваемых параметров выбраны epoch = 1, 2, 5, 7, 10; lr - 0.001, 0.01, 0.1, 0.5, 1. Наиболее высокая точность достигнута при lr = 0.001 и epoch = 1:
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
Как я поняла из документации, для регрессии другие указанные метрики (например, confusion matrix) не используются.  
2) Метод опорных векторов  
Модель: <img src="https://render.githubusercontent.com/render/math?math=class(x)=sign(X*w)">   
  
Нам также необходимо настроить значения w и b. Гиперпараметры - количество итераций и скорость обучения.  
Во время обучения сначала мы строим вектор, определяющий, какой класс у объекта:  
```
y_ = np.where(y > 0, 1, -1)
```
Инициализируем w - нулевой. Далее проходим по всем эпохам, вычисляя condition. Вычисляем значение функции потерь, затем, если ее значение меньше 1, то:  
<img src="https://render.githubusercontent.com/render/math?math=w=lr*(x[i]*y[i]-2/epoch*w) + w">     
иначе:  
<img src="https://render.githubusercontent.com/render/math?math=w=w+lr*(-2)/epoch*w">   
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
3) Метод ближайших соседей  
Здесь мы берем k ближайших к рассматриваемому объекту соседей (их значения уже известны) и смотрим, какой из них наиболее часто встречается, и присваиваем этому объекту данный класс.  
Поэтому в обучении в данном методе просто запоминаются тренировочные входная и выходная выборки. Гиперпараметры - число соседей и функция, с помощью которой будем измерять расстояние. У меня их 2: евклидово и абсолютное расстояние.  

