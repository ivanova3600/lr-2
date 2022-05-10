# Лабораторная работа №1

### Введение
В ходе данной лабораторной работы необходимо реализовать следующие алгоритмы машинного обучения: Linear/ Logistic Regression, SVM, KNN, Naive Bayes в отдельных классах. Также нужно настроить параметры моделей с помощью GridSeachSV или RandomSearchCV, организовать весь процесс с помощью Pipeline. Аналогично проделать с помощью коробочных решений. Также оценить полученные модели.  
### Ход работы
1) Линейная регрессия (с градиентным спуском)  
Гиперпараметры - lr (скорость обучения) и epoch (количество эпох).  
Модель: <img src="https://render.githubusercontent.com/render/math?math=X*w + b = Y">

weight - веса, b - шум, их инициализируем нулями. Во время обучения мы подбираем такие w и b, чтобы модель давала правильные ответы: сначала мы вычисляем y_pred по формуле выше. Далее мы вычисляем функцию потерь: <img src="https://render.githubusercontent.com/render/math?math=1/(2*n)*\sum_{i=0}^{n} (y_{pred}-y)^2">  
```
np.sum(np.square(y_pred-y))/(2*self.m)
```
self.m - длина входного массива, y - выходные точные данные, y_pred - предсказанные выходные данные. Далее находим градиенты:  
<img src="https://render.githubusercontent.com/render/math?math=dw = X^T*(y_{y_pred-y)/n">  
<img src="https://render.githubusercontent.com/render/math?math= db = 1/2*\sum_{i=0}^{n} (y_{pred}-y)">  
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
