
import pandas as pd

#数据加载
train_data=pd.read_csv("./Titanic_Data-master/train.csv")
test_data=pd.read_csv("./Titanic_Data-master/test.csv")

#数据探索
print(train_data.info())
print('-'*30)
print(train_data.describe())
print('-'*30)
print(train_data.describe(include=['O']))
print('-'*30)
print(train_data.head())
print('-'*30)
print(train_data.tail())

'''
通过数据探索，我们发现 Age、Fare 和 Cabin 这三个字段的数据有所缺失。
其中 Age 为 年龄字段，是数值型，我们可以通过平均值进行补齐；
Fare 为船票价格，是数值型，
我们 也可以通过其他人购买船票的平均值进行补齐。
'''
#使用平均值来填充年龄为nan的值
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
#使用股票的均值填充票价中的nan
train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

'''
Cabin 为船舱，有大量的缺失值。在训练集和测试集中的缺失率分别为 77% 和 78%，无 法补齐；
Embarked 为登陆港口，有少量的缺失值，我们可以把缺失值补齐。
'''
#观察Embarked字段取值
#我们发现一共就 3 个登陆港口，其中 S 港口人数最多，占到了 72%，因此我们将其余缺 失的 Embarked 数值均设置为 S：
print(train_data['Embarked'].value_counts())
# 使用登录最多的港口来填充登录港口的 nan 值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

#特征选择 寻找自认为可能会和乘客的预测分类有关系的特征
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
#选择特定列
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

#特征值中是字符串 不方便处理 用数字来代替比如female为1 male为0 Embarked为S C Q三种可能 用0/1表示
from sklearn.feature_extraction import DictVectorizer
dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))#fit_transform它可以将特征向量转化为特征值矩阵
print(dvec.feature_names_)
#clf = DecisionTreeClassifier(criterion='entropy')
from sklearn.tree import DecisionTreeClassifier # 构造 ID3 决策树
clf = DecisionTreeClassifier(criterion='entropy') # 决策树训练
clf.fit(train_features, train_labels)
test_features=dvec.transform(test_features.to_dict(orient='record'))
# 决策树预测
pred_labels = clf.predict(test_features)
# 得到决策树准确率
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(u'score 准确率为 %.4lf' % acc_decision_tree)
