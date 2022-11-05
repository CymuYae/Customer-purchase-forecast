#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 代码参考 “泰坦尼克号生存者”修改编写而成
# @author CymuYae
# github https://github.com/CymuYae


# In[198]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[199]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[200]:


get_ipython().system('pip install seaborn -i https://mirrors.aliyun.com/pypi/simple')
# 这里不整成1.2.0算法6跑不动
get_ipython().system('pip install xgboost==1.2.0')


# In[201]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[202]:


# 导入库
import numpy as np
import pandas as pd
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 导入数据
train = pd.read_csv('data/data175160/train.csv')
test = pd.read_csv('data/data175160/test.csv')
display(train.head(n=1),test.head(n=1))


# In[203]:


# 查看数据信息
train.info()
test.info()


# In[204]:


#为了方便查看处理，把训练集、数据集合在一起,通过id可以把两个数据集再分开
con=pd.concat([train,test]).set_index('id')
con


# In[205]:


#查看每一列有多少unknown，太多的不用力
con.replace('unknown',np.nan,inplace=True)
num = con.isna().sum()
num


# In[ ]:





# In[206]:


# 采用seaborn绘图函数库作可视化分析
# 职业
sns.countplot(x="job", hue="subscribe", data=train)
plt.xticks(rotation=45)


# In[207]:


#将职业特征分类
# ['admin. = 0','services = 1','blue-collar = 2','entrepreneur = 3','management = 4','technician = 5',
# 'housemaid = 6','self-employed = 7','unemployed = 8','retired = 9','student = 10','unknown = 11']
le =LabelEncoder()
le.fit(['admin.','services','blue-collar','entrepreneur','management','technician','housemaid','self-employed','unemployed','retired','student','unknown'])
train['job']=le.transform(train['job'])
print(train['job'])


# In[208]:


#职业特征分类，退休工人、学生最多，蓝领最低
train['job']=train['job'].map(lambda x:'high' if 9<=x<=10 else 'mid' if x==8 else "low")
# 作图比较
sns.countplot(x="job", hue="subscribe", data=train)


# In[209]:


# 是否有房贷
sns.countplot(x="housing", hue="subscribe", data=train)
plt.xticks(rotation=45)


# In[210]:


#信用卡是否违约
sns.countplot(x="default", hue="subscribe", data=train)
plt.xticks(rotation=45)


# In[211]:


# 是否有个人贷款
sns.countplot(x="loan", hue="subscribe", data=train)
plt.xticks(rotation=45)


# In[212]:


# 信用卡是否有违约
sns.countplot(x="default", hue="subscribe", data=train)
plt.xticks(rotation=45)


# In[213]:


# 就业变动率
sns.countplot(x="emp_var_rate", hue="subscribe", data=train)
plt.xticks(rotation=45)


# In[214]:


sns.violinplot(x='subscribe',y='emp_var_rate',data=train)


# In[215]:


# 就业变动率特征
train['emp_var_rate']=train['emp_var_rate'].map(lambda x: 'high' if x<-0.5 else 'low' if x<2 else 'toohigh' if x>=2 else 'null')


# In[216]:


#将有房贷的转化为yes,缺损的转化为no
train['housing']=train['housing'].map(lambda x:'yes' if x!="unknown"  else 'no')
# 作图比较
sns.countplot(x="housing", hue="subscribe", data=train)


# In[217]:


# 将有个人贷款的转化成yes，缺损的转化成no
train['loan']=train['loan'].map(lambda x:'yes' if x!="unknown"  else 'no')
#作图比较
sns.countplot(x="loan", hue="subscribe", data=train)


# In[218]:


# 将有信用卡违约的转化成yes，缺损的转化成no
train['default']=train['default'].map(lambda x:'yes' if x!="unknown"  else 'no')
#作图比较
sns.countplot(x="default", hue="subscribe", data=train)


# In[219]:


#年龄小提琴图
sns.violinplot(x='subscribe',y='age',data=train)


# In[220]:


#年龄特征分类
train['age']=train['age'].map(lambda x: 'youth' if x<25 else 'adlut' if x<55 else 'old' if x<95 else 'tooold' if x>=95 else 'null')


# In[221]:


# 年龄
sns.countplot(x="age", hue="subscribe", data=train)
plt.xticks(rotation=45)


# In[222]:


#联系方式
sns.countplot(x="contact", hue="subscribe", data=train)
plt.xticks(rotation=45)


# In[223]:


le =LabelEncoder()
le.fit(['no','yes'])
train['subscribe']=le.transform(train['subscribe'])
print(train['subscribe'])


# In[224]:


#删掉含有缺损值的样本
train.dropna(axis=0,inplace=True)
#查看训练集的信息
train.info()


# In[225]:


#将训练数据分成标记和特征两部分
labels= train['subscribe']
features= train.drop(['subscribe','id','marital','education','month','day_of_week','duration','campaign','pdays','previous','poutcome','cons_price_index','cons_conf_index','lending_rate3m','nr_employed'],axis=1)


# In[226]:


#对所有特征实现独热编码
features = pd.get_dummies(features)
encoded = list(features.columns)
print ("{} total features after one-hot encoding.".format(len(encoded)))
features.info()


# In[227]:


#将职业特征分类
# ['admin. = 0','services = 1','blue-collar = 2','entrepreneur = 3','management = 4','technician = 5',
# 'housemaid = 6','self-employed = 7','unemployed = 8','retired = 9','student = 10','unknown = 11']
le =LabelEncoder()
le.fit(['admin.','services','blue-collar','entrepreneur','management','technician','housemaid','self-employed','unemployed','retired','student','unknown'])
test['job']=le.transform(test['job'])
print(test['job'])


# In[228]:


# 分段分类
test['age']=test['age'].map(lambda x: 'youth' if x<25 else 'adlut' if x<55 else 'old' if x<95 else 'tooold' if x>=95 else 'null')
test['emp_var_rate']=test['emp_var_rate'].map(lambda x: 'low' if x<-0.5 else 'high' if x<2 else 'toohigh' if x>=2 else 'null')
test['housing']=test['housing'].map(lambda x:'yes' if x!="unknown"  else 'no')
test['default']=test['default'].map(lambda x:'yes' if x!="unknown"  else 'no')
test['loan']=test['loan'].map(lambda x:'yes' if x!="unknown"  else 'no')
test['job']=test['job'].map(lambda x:'high' if 9<=x<=10 else 'mid' if x==8 else "low")
#删除不需要的特征并进行独热编码
id = test['id']
test = test.drop(['id','marital','education','month','day_of_week','duration','campaign','pdays','previous','poutcome','cons_price_index','cons_conf_index','lending_rate3m','nr_employed'],axis=1)
test=pd.get_dummies(test)
encoded = list(test.columns)
print ("{} total features after one-hot encoding.".format(len(encoded)))
test.info()


# In[229]:


#首先引入需要的库和函数
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,roc_auc_score
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier


# In[230]:


#定义通用函数框架
def fit_model(alg,parameters):
    X=np.array(features)
    y=np.array(labels)
    scorer=make_scorer(roc_auc_score)  #使用roc_auc_score作为评分标准
    grid = GridSearchCV(alg,parameters,scoring=scorer,cv=5)  #使用网格搜索，出入参数
    start=time()  #计时
    grid=grid.fit(X,y)  #模型训练
    end=time()
    t=round(end-start,3)
    print (grid.best_params_)  #输出最佳参数
    print ('searching time for {} is {} s'.format(alg.__class__.__name__,t)) #输出搜索时间
    return grid #返回训练好的模型


# In[231]:


#列出需要使用的算法
alg1=DecisionTreeClassifier(random_state=29)
alg2=SVC(probability=True,random_state=29)  #由于使用roc_auc_score作为评分标准，需将SVC中的probability参数设置为True
alg3=RandomForestClassifier(random_state=29)
alg4=AdaBoostClassifier(random_state=29)
alg5=KNeighborsClassifier(n_jobs=-1)
alg6=XGBClassifier(random_state=29,n_jobs=-1)


# In[232]:


#列出需要调整的参数范围
parameters1={'max_depth':range(1,10),'min_samples_split':range(2,10)}
parameters2 = {"C":range(1,20), "gamma": [0.05,0.1,0.15,0.2,0.25]}
parameters3_1 = {'n_estimators':range(10,200,10)}
parameters3_2 = {'max_depth':range(1,10),'min_samples_split':range(2,10)}  #搜索空间太大，分两次调整参数
parameters4 = {'n_estimators':range(10,200,10),'learning_rate':[i/10.0 for i in range(5,15)]}
parameters5 = {'n_neighbors':range(1,10),'leaf_size':range(10,60,20)  }
parameters6_1 = {'n_estimators':range(10,200,10)}
parameters6_2 = {'max_depth':range(1,10),'min_child_weight':range(1,10)}
parameters6_3 = {'subsample':[i/10.0 for i in range(1,10)], 'colsample_bytree':[i/10.0 for i in range(1,10)]}#搜索空间太大，分三次调整参数


# In[233]:


def save(clf,i):
    pred=clf.predict(np.array(test))
    sub=pd.DataFrame({ 'id': id, 'subscribe': pred })
    sub.to_csv("res_{}.csv".format(i), index=False)


# In[234]:


clf1=fit_model(alg1,parameters1)


# In[236]:


#跑的慢
clf2=fit_model(alg2,parameters2)


# In[ ]:


clf3_m1=fit_model(alg3,parameters3_1)


# In[195]:


alg3=RandomForestClassifier(random_state=29,n_estimators=180)
clf3=fit_model(alg3,parameters3_2)


# In[ ]:


clf4=fit_model(alg4,parameters4)


# In[196]:


clf5=fit_model(alg5,parameters5)


# In[ ]:


clf6_m1=fit_model(alg6,parameters6_1)


# In[ ]:


alg6=XGBClassifier(n_estimators=140,random_state=29,n_jobs=-1)
clf6_m2=fit_model(alg6,parameters6_2)


# In[ ]:


alg6=XGBClassifier(n_estimators=140,max_depth=4,min_child_weight=5,random_state=29,n_jobs=-1)
clf6=fit_model(alg6,parameters6_3)


# In[ ]:


i=1
for clf in [clf1,clf2,clf3clf4,clf5,clf6]:
   save(clf,i)
   i=i+1

