#!/usr/bin/env python
# coding: utf-8

# ---
# # Python 人工智能入门 毕业项目
# 
# # 利用机器学习模型预测股票走势
# 
# ## Yunqing Bao
# ## 2019年6月30日
# ---
# 

# ## 目录
# <ul>
# <li><a href="#definition">定义</a></li>
# <li><a href="#analysis">分析</a></li>
# <li><a href="#implementation">实现</a></li>
# <li><a href="#result">结果</a></li>
# <li><a href="#conclusion">结论</a></li>   
# </ul>
# 
# 

# ---
# <a id="definition"></a>
# ## Ⅰ.定义
# 
# 在第一节中，你需要对你选定的问题作出定义
# 
# ### 1.项目概况
# 
# 在这个部分，你需要用浅显简洁的语句描述这个项目的一个总体概况。以下几个问题可以帮助你理清思路：
# 
# - _项目的背景信息是什么？_
# - _做这个项目的出发点？_
# - _数据集的大概情况是什么？_
# 
# ### 2.问题陈述
# 
# 在这个部分，你需要清楚地为你将要解决的问题下定义，这应该包括你解决问题将要使用的策略（任务的大纲）。你同时要详尽地讨论你期望的结果是怎样的。有几个问题是需要考虑的：
# - _你是否清楚地定义了这个问题。站在读者的角度，他们能否明白你将要解决的问题是什么。_
# - _你是否详尽地阐述了你将会如何解决这个问题？_
# - _你期望什么样的结果，读者能明白你期望的这个结果吗？_
# 
# ### 3.评价指标
# 在这里，你需要说明你将要用于评价自己的模型和结果的**指标**和**计算方法**。它们需要契合你所选问题的特点及其所在的领域，同时，你要保证他们的合理性。需要考虑的问题：
# - _你是否清晰地定义了你所使用的指标和计算方法？_
# - _你是否论述了这些指标和计算方法的合理性？_

# 我们获取了一定时间内的股票大盘数据。 
# 通过这些大盘数据，我们希望预测一支股票的走势。
# 根据其过去的涨跌走势，预测出其未来的涨跌走势情况。
# 
# 数据集中包含了交易的日期、当天的最高与最低指数，收盘指数和每天的交易额
# 数据集中还包含了CHG与CHGpct指数（显示每天股表指数的涨跌和相应的百分比）
# 
# 通过这些数据，我希望能够做到预测一段时间内股票的走势，股票指数所能达到的最高点与最低点。
# 读者们为了利益的最大化，将会被期望在预测指数最低点当天买入，在预测指数最高点当天卖出。
# 
# 为了评价预测模型的结果好坏，我们将一部分的数据集当作训练集，另一部分的数据集当作测试集。
# 将从训练集中得到的预测模型放入测试集中进行检验，使用线性回归的方法进行预测。
# p值可以体现变量的影响程度，p值很接近于，说明变量对预测量有较大的线性关系影响。
# 
# MSE，为预测值和测试值之间的平均误差的方差，MSE越小，说明模型预测的贴近实际值。
# R2，为决定系数，表示了拟合程度的好坏。越接近1，表明方程的变量对y的解释能力越强，这个模型对数据拟合的也较好
# 越接近0，表明模型拟合的越差。经验值：R2>0.4， 则拟合效果好
# 
# 

# ---
# <a id="analysis"></a>
# ## Ⅱ.分析
# 
# 在第二节，你将对数据进行分析与整理，来进一步认识数据。
# 
# ### 1.数据的探索
# 在这一部分，你需要探索你将要使用的数据。数据可以是若干个数据集，或者输入数据/文件，甚至可以是一个设定环境。不过在本项目中，我们已经给出了数据集，因此，你需要详尽地描述数据的类型。如果可以的话，你需要展示数据的一些统计量和基本信息（例如输入的特征（features)，输入里与定义相关的特性）。你还要说明数据中的任何需要被关注的异常或有趣的性质（例如需要做变换的特征，离群值等等）。你需要考虑：
# - _如果你使用了数据集，你要详尽地讨论了你所使用数据集的某些特征，并且为阅读者呈现一个直观的样本_
# - _如果你使用了数据集，你要计算并描述了它们的统计量，并对其中与你问题相关的地方进行讨论_
# - _数据集或输入中是否存在异常、缺陷或其他特性？你为什么认为他们是异常？给出佐证你观点的理由(例如分类变量的处理，缺失数据，离群值等）_
# 
# ### 2.探索性可视化
# 在这一部分，你需要对数据的特征或特性进行概括性或提取性的可视化。这个可视化的过程应该要适应你所使用的数据。就你为何使用这个形式的可视化，以及这个可视化过程为什么是有意义的，进行一定的讨论。你需要考虑的问题：
# - _你是否对数据中与问题有关的特性进行了可视化？_
# - _你对可视化结果进行详尽的分析和讨论了吗？_
# - _绘图的坐标轴，标题，基准面是不是清晰定义了？_
# 
# ### 3.算法和技术
# 在这一部分，你需要讨论你解决问题时用到的算法和技术。你需要根据问题的特性和所属领域来论述使用这些方法的合理性。你需要考虑：
# - _你所使用的算法，包括用到的变量/参数都清晰地说明了吗？_
# - _你是否已经详尽地描述并讨论了使用这些技术的合理性？_

# > **提示**：*不应* 在每个 notebook 框 (cell) 中进行太多操作。可以自由创建框，来进行数据探索。在这个项目中，可以在初始 notebook 中进行大量探索性操作。不要求对其进行组织，但请务必仔细阅读备注，理解每个代码框的用途。完成分析之后，你可以创建 notebook 副本，在其中去除多余数据，组织好你的每一步分析，从而形成信息连贯、结构紧密的报告。

# In[287]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from patsy import dmatrices
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, svm
from sklearn.metrics import mean_squared_error, r2_score
import math
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('data.csv',index_col= 0)
df.head()


# In[288]:


df.corr()


# In[289]:


df.describe()


# **(你的回答)**
# 一共有2187组数据，没有缺失值。
# tradeDate表示交易的日期。
# closeindex表示每日的收盘指数。
# highestindex和lowestindex表示每日的最高和最低股票指数。
# turnoverVol表示每天的交易额。
# CHG和CHGPct表示每天股票指数的涨跌幅度和相应的百分比。
# 
# CHG的平均值为1.152,表示从2006年至2015年2月股票指数总体上涨
# CHG的标准差值比较大，表示股票指数的波动性较强。
# 

# In[290]:


df.plot(x = 'tradeDate',y = 'CHG')


# In[291]:


df.plot(x = 'tradeDate',y = 'closeIndex')


# In[292]:


plt.plot(df['highestIndex'],color='blue',label = 'highestIndex')
plt.plot(df['lowestIndex'],color='red', label = 'lowestIndex')
plt.plot(df['closeIndex'],color='green', label = 'closeIndex')
plt.legend()


# In[293]:


y, X = dmatrices('closeIndex ~ highestIndex + lowestIndex + turnoverVol',df,return_type = 'dataframe' )

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif


# In[294]:


y, X = dmatrices('closeIndex ~ highestIndex + turnoverVol',df,return_type = 'dataframe' )

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif


# In[295]:


从图中可以看出，股票的收盘指数和每天的最高指数、最低指数的变化趋势是一致的。
从VIF系数看来，closeIndex、highestIndex与lowIndex之间存在多重共线性。去除一个lowestIndex后，VIF小于10，是合格的。
在一天之内指数的变动程度要比股票指数随不同日期变化的程度小的多。
从预测股票走势的角度来看，highestIndex,lowestIndex与closeIndex都可以作为预测的目标，他们都可以比较好的反应股票的走势。


# ---
# <a id="implementation"></a>
# ## Ⅲ. 实现
# 在第三节中，你需要引入自己选定的模型，并使用数据集中的数据去训练兵优化它，使得最终的模型能够得到想要的结果
# 
# ### 1.数据预处理
# 在这一部分， 你需要清晰记录你所有必要的数据预处理步骤。在前一个部分所描述的数据的异常或特性在这一部分需要被更正和处理。需要考虑的问题有：
# - _如果你选择的算法需要进行特征选取或特征变换，你对此进行记录和描述了吗？_
# - _**数据的探索**这一部分中提及的异常和特性是否被更正了，对此进行记录和描述了吗？_
# - _如果你认为不需要进行预处理，你解释个中原因了吗？_
# 
# ### 2.执行过程
# 在这一部分， 你需要描述你所建立的模型在给定数据上执行过程。模型的执行过程，以及过程中遇到的困难的描述应该清晰明了地记录和描述。需要考虑的问题：
# - _你所用到的算法和技术执行的方式是否清晰记录了？_
# - _在运用上面所提及的技术及指标的执行过程中是否遇到了困难，是否需要作出改动来得到想要的结果？_
# 
# 
# ### 3.完善
# 在这一部分，你需要描述你对原有的算法和技术完善的过程。例如调整模型的参数以达到更好的结果的过程应该有所记录。你需要记录最初和最终的模型，以及过程中有代表性意义的结果。你需要考虑的问题：
# - _初始结果是否清晰记录了？_
# - _完善的过程是否清晰记录了，其中使用了什么技术？_
# - _完善过程中的结果以及最终结果是否清晰记录了？_

# In[296]:


#观察CHG对closeIndex的影响
df['intercept'] = 1
lm = sm.OLS(df['closeIndex'],df[['intercept','CHG','CHGPct']])
result = lm.fit()
result.summary()


# In[297]:


#观察CHG,turnoverVol对closeIndex的影响
df['intercept'] = 1
lm = sm.OLS(df['closeIndex'],df[['intercept','turnoverVol','CHG','CHGPct']])
result = lm.fit()
result.summary()


# In[298]:


#预测forecast_out天后的
forecast_out = 20

df_new = df.copy()
df_new = df.drop(df.columns[[0,5,6]],axis = 1)

print(df_new)


df_new['lable'] = df_new['closeIndex'].shift(-forecast_out)

X = np.array(df_new.drop(['lable'], axis = 1))

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df_new.dropna(inplace = True)
y = np.array(df_new['lable'])


# **(你的回答)**
# 若是去掉CHG指数等变量，还是能够很好的预测股票收盘指数，收盘指数closeindex约99.9%与highestindex、lowestindex和turnoverVol有关。
# 
# 考虑到closeIndex与highestIndex和lowestIndex之间存在较大的多重共线性，所以可以除去lowestIndex
# 
# 在线性模型中，closeindex与CHG,CHGPct这两个变量的R平方系数只有0.002,说明关系不大，可以忽略它们的影响。
# 在加入turnoverVol变量后，R平方系数增大为0.025，说明turnoverVol对收盘指数还是有一定影响的，不能被忽视。
# 
# 去掉空值后，可以预测出forecast_out天后的closeindex值，用于预测股票指数的走势。

# ---
# <a id="result"></a>
# ## IV. 结果
# 经过前面的几步，你已经训练好了自己的模型并计算出了一些结果。这一节，你需要对这些进行讨论与分析
# 
# ### 模型的评价与验证
# 在这一部分，你需要对你得出的最终模型的各种技术质量进行详尽的评价。最终模型是怎么得出来的，为什么它会被选为最佳需要清晰地描述。你也需要对模型和结果可靠性作出验证分析，譬如对输入数据或环境的一些操控是否会对结果产生影响（敏感性分析sensitivity analysis）。一些需要考虑的问题：
# - _最终的模型是否合理，跟期待的结果是否一致？最后的各种参数是否合理？_
# - _模型是否对于这个问题是否足够稳健可靠？训练数据或输入的一些微小的改变是否会极大影响结果？（鲁棒性）_
# - _这个模型得出的结果是否可信？_
# 
# 

# In[305]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=0)
lm = LinearRegression()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_lately)


#将预测结果转化为字典，保存未来20天的股票指数
last_date = df.iloc[-1].name
x_day = np.linspace(1,forecast_out,forecast_out)

a = np.array(x_day)
b = np.array(y_pred)
b = np.append(b,np.array(y_pred))
predict = dict(zip(a,b))
print('predict:{A}'.format(A=predict))


# In[307]:


#对预测结果进行评价，使用MSE与R2两个评价系数
df_new = df.copy()
df_new = df.drop(df.columns[[0,2,5,6]],axis = 1)

X_data = df_new.drop('closeIndex', axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(
                 X_data, df_new['closeIndex'], test_size=0.1, random_state=0)
lm.fit(X_train, y_train)
y_pred_2 = lm.predict(X_test)

# 查看残差平方的均值(mean square error,MSE) 
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred_2))

# R2 决定系数（拟合优度）
# 模型越好：r2→1
# 模型越差：r2→0
print('Variance score: %.4f' % r2_score(y_test,y_pred_2))


# **(你的回答)**
# 模型主要从highest、lowestindex和closeindex三个数据进行预测，得出了未来20天的股票指数的走势。
# 
# 在评价后，发现均方差的值为1262.51,相对于股票指数平均几千的值来说，已经比较小了。
# 而系数R2表示了数据的拟合程度，0.9987的值很接近于1,是比较理想的。
# 从上述两个评价系数的表现，得知股票的走势的结果还是比较可信的。
# 

# ---
# <a id="conclusion"></a>
# ## V. 项目结论
# 这一节中，我们将对整个项目做出总结
# 
# 
# ### 结果可视化
# 在这一部分，你需要用可视化的方式展示项目中需要强调的重要技术特性。至于什么形式，你可以自由把握，但需要表达出一个关于这个项目重要的结论和特点，并对此作出讨论。一些需要考虑的：
# - _你是否对一个与问题，数据集，输入数据，或结果相关的，重要的技术特性进行了可视化？_
# - _可视化结果是否详尽的分析讨论了？_
# - _绘图的坐标轴，标题，基准面是不是清晰定义了？_
# 
# 
# ### 对项目的思考
# 在这一部分，你需要从头到尾总结一下整个问题的解决方案，讨论其中你认为有趣或困难的地方。从整体来反思一下整个项目，确保自己对整个流程是明确掌握的。需要考虑：
# - _你是否详尽总结了项目的整个流程？_
# - _项目里有哪些比较有意思的地方？_
# - _项目里有哪些比较困难的地方？_
# - _最终模型和结果是否符合你对这个问题的期望？它可以在通用的场景下解决这些类型的问题吗？_
# 
# 
# ### 需要作出的改进
# 在这一部分，你需要讨论你可以怎么样去完善你执行流程中的某一方面。例如考虑一下你的操作的方法是否可以进一步推广，泛化，有没有需要作出变更的地方。你并不需要确实作出这些改进，不过你应能够讨论这些改进可能对结果的影响，并与现有结果进行比较。一些需要考虑的问题：
# - _是否可以有算法和技术层面的进一步的完善？_
# - _是否有一些你了解到，但是你还没能够实践的算法和技术？_
# 
# 
# 

# In[308]:


#画出未来有20天中收盘指数的预期变化
plt.figure(figsize=(8, 6.5))
plt.plot(x_day,y_pred)
plt.scatter(x_day,y_pred)
plt.xlabel('day')
plt.ylabel('closeIndex')           
plt.show()
# 你可以自由的使用任意数量和任意格式的代码框，但请在最终提交的报告中注意报告的整洁与通顺


# **(你的回答)**
# 画出了收盘指数在未来有20天中的走势，以折线图与散点图的方式。
# 从图中可以看出，第10天的股票指数最低，而在之后第19天的收盘指数最高。
# 所以应该在第10天买入，19天卖出。
# 
# 这次模型预测主要用了每天最高与收盘的股票指数与交易额去预测未来每天股票的收盘指数。
# 而CHG指数对closeindex的影响很小，可以不考虑。每天的收盘指数与最高和最低指数的数值是差不多的，这三者的角色也是
# 基本相同的，所以这次的预测主要还是用交易额与收盘指数去预测未来的股票指数。
# 
# 但现实中能影响股票的变量是非常多的，单纯数字上的收盘指数和交易额并不足够预测出可靠的股票走势。还有很多诸如政策影响、主力资金流动、股民的信心等等变量会对股票走势产生很大的影响。我们需要更多的数据与变量去做菜更加精确、可靠的预测。

# ----------
# ** 在提交之前， 问一下自己... **
# 
# - 你所写的项目报告结构对比于这个模板而言足够清晰了没有？
# - 每一个部分（尤其**分析**和**方法**）是否清晰，简洁，明了？有没有存在歧义的术语和用语需要进一步说明的？
# - 你的目标读者是不是能够明白你的分析，方法和结果？
# - 报告里面是否有语法错误或拼写错误？
# - 报告里提到的一些外部资料及来源是不是都正确引述或引用了？
# - 代码可读性是否良好？必要的注释是否加上了？
# - 代码是否可以顺利运行并重现跟报告相似的结果？
