# 2023-数据集成-路线二第三次作业报告

[toc]

## 1. 基本信息

**小组编号：4**

| 姓名   | 学号      | 分工               |
| ------ | --------- |:------------------ |
| 万沛沛 | 201250038 | 组长、信用等级预测 |
| 邓尤亮 | 201250035 | 星级预测           |
| 韩陈旭 | 201250037 | 信用等级预测       |
| 张月明 | 201830115 | 信用等级预测       |
| 华广松 | 201840309 | 星级预测           |

## 2. 项目文件

项目仓库：[https://github.com/HCPlantern/2023-Data-Integration-hw3](https://github.com/HCPlantern/2023-Data-Integration-hw3)

```
2023-Data-Integration-hw3
├── code                                
│   ├── credit                        // 信用预测相关
│   │   ├── confusion_matrix_xgb.png    // 混淆图
│   │   └── xgboost.joblib              // 模型
│   ├── credit_preditc.ipynb            // 代码
│   ├── star                          // 星级预测相关
│   │   ├── confusion_matrix_xgb.png    // 混淆图
│   │   └── xgboost.joblib              // 模型
│   └── star_predict.ipynb              // 代码
├── README.md                       // 说明文档
└── result                          // 最终的预测结果
    ├── credit_result.csv
    └── star_result.csv
```

该项目仅包含了相关代码和结果文件，具体使用的CSV文件和数据由于体积过大，存放在了 NJUBOX 中：[2023-Data-Integration-hw3-zip](https://box.nju.edu.cn/f/bbb76061527b432184d3/)

该项目所用到的数据来源于 hw2 中的 Clickhouse 数据库。我们选择出合适的字段后通过 SQL 语句导出成 CSV 文件再交给 Python 进行数据处理和模型训练。

## 3. 项目说明

该部分按照评分标准详细阐述团队的探索过程，说明我们尝试过的方法、具体的步骤以及代码示例。其中，有些方法经过尝试之后并没有在最终的模型训练和预测的结果中使用。

我们使用了 JetBrains 的 Datalore 用于团队合作。这是一个在线 Juypter Notebook 平台，支持多人协作。地址：[Datalore](https://datalore.jetbrains.com/notebook/CQdWw16N0v5MwggDtoZGS5/IqzAj4BmGrup6CDhcPDuIF) 

由于在线平台的机能受限，最终训练模型时我们在本地环境运行代码，提高效率。

### 3.1 数据收集

星级预测和信用预测的字段选择说明、数据盘点和可视化将在 **星级预测** 和 **信用预测** 部分详细说明。

数据盘点的相关代码如下：

```python
import seaborn as sns
import matplotlib.pyplot as plt
columns = all_data.columns.to_list()
columns.remove('uid')
columns.remove('star_level')
columns = [col for col in all_data.columns if col in columns and all_data[col].dtype == np.number]

fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(15, 6))
for i, columns in enumerate(columns):
    sns.boxplot(data=all_data[[columns]], ax=axes[i])
plt.tight_layout()
plt.savefig(PREFIX + 'boxplot.png', dpi=300)

```

### 3.2 数据预处理

该部分涉及到数据的缺失值处理、异常值处理、极端值处理和数据标准化等部分。

#### 3.2.1 缺失值处理

该部分我们的处理是去除了缺失值比例大于 0.7 的列，并用中位数填充数值型变量、用众数填充类别型变量缺失值。

```python
# 计算文件每一列的缺失值比例并保存至 missing.csv 文件
all_data.isnull().mean().to_csv(missing_name)
# 去除缺失值比例大于 0.7 的列
all_data = all_data.loc[:, all_data.isnull().mean() < 0.7]
#对于数值型变量，用中位数填充缺失值
all_data.fillna(all_data.median(numeric_only=True), inplace=True)
#对于类别型变量，用众数填充缺失值
all_data.fillna(all_data.mode().iloc[0], inplace=True)
print(all_data.columns)
print(all_data.dtypes)

```

#### 3.2.2 异常值和极端值处理

异常值可以理解为超出正常数据范围的数据，例如身高、体重异常。但由于我们的字段选择大多数为存款等信息，这种金额类型的字段只要不是负数都可以被认为是合理的。因此我们并没有作出处理。

至于极端值的处理，我们考虑过使用 z-score 来处理存款这类连续型数据，并使用盖帽法替换极端值。

```python
# numlist 中是需要进行计算 z-score 的列名列表
for col_name in numlist:
    # 计算每列的 z-score
    zscore = stats.zscore(all_data[col_name])
    p99 = all_data[col_name].quantile(0.99)
    p1 = all_data[col_name].quantile(0.01)
    # 将 z-score 大于 3 的值视为异常值并进行处理，使用盖帽法将其替换为 99% 或 1% 的数值
    all_data[col_name] = np.where(zscore > 3, p99, all_data[col_name])
    all_data[col_name] = np.where(zscore < -3, p1, all_data[col_name])
all_data.describe()
```

不过，由于我们最终选择了 xgboost 模型进行训练。基于决策树模型，xgboost 并不需要处理极端值，反而可能会导致有用的信息被删除。因此该极端值处理方法在最终的数据处理时并没有被使用。

#### 3.2.3 数据转换和标准化


由于部分字段为类别型变量，因此需要进行数据转换。我们尝试了 k 值编码和 One-Hot 编码，并发现使用了 One-Hot 编码的 xgboost 模型准确率有所提升，因此最终的模型训练使用了 One-Hot 编码。

具体代码如下：

k值编码：
```python
le = LabelEncoder()
all_data[catelist] = all_data[catelist].apply(le.fit_transform)
```

One-Hot 编码：
```python
all_data = pd.get_dummies(all_data, columns=catelist)
```

至于数据标准化部分则较为简单：

```python
scaler = StandardScaler()
all_data[columns] = scaler.fit_transform(all_data[columns])
```

不过，由于 xgboost 并不要求数据标准化或归一化，所以最终的训练中没有进行标准化操作。

#### 3.2.4 其他处理

在数据盘点时，我们观察到部分字段的 0 值比例过大，甚至全为 0。对于全 0 字段我们进行了字段删除；而对于 0 值比例过大但不全为 0 的字段，考虑到这些信息对于分类预测有着比较重要的意义，因此并没有删除。

例如，星级预测中被选择的字段 `avg_qur`, `td_crd_bal` 和 `oth_td_bal` 由于数据全 0 而被删除。

### 3.3 特征工程

该部分包含线性相关性和多重共线性的处理。此外，对于 xgboost 模型，我们还尝试了使用 PCA 消除多重共线性。

#### 3.3.1 线性相关性

我们尝试了以下代码以计算列的相关性，并使用前向删除法删除相关性较高的列。

```python
def forward_delete_corr(data, columns, threshold):
    """
    前向删除法：根据变量之间的相关性，逐步删除相关性较高的变量
    data: 数据集
    columns: 变量列表
    threshold: 相关性阈值，超过此阈值的变量将被删除
    """
    # 计算相关性矩阵
    corr_matrix = data[columns].corr()
    # 初始化待删除变量列表
    delete_columns = []
    # 逐个遍历变量，计算其与其他变量的相关系数
    for i, column in enumerate(columns[:-1]):
        for j in range(i + 1, len(columns)):
            if abs(corr_matrix.loc[column, columns[j]]) >= threshold:
                # 如果相关系数超过阈值，将其加入待删除变量列表
                delete_columns.append(columns[j])
    # 对 delete_columns 去重
    delete_columns = list(set(delete_columns))
    # 将待删除变量从变量列表中删除
    columns = [column for column in columns if column not in delete_columns]
    print("delete_columns: ", delete_columns)
    return columns

columns = forward_delete_corr(all_data[columns], columns, 0.7)
```

#### 3.3.2 多重共线性

我们尝试了以下代码用于处理多重共线性。

```python
def get_low_vif_cols(dataframe, thresh=5.0):
    """
    基于 VIF 值筛选多重共线性较小的特征列
    dataframe: DataFrame，需要进行 VIF 值计算的数据集
    thresh: float，VIF 值的阈值，默认为 5.0
    返回值：List，筛选出的特征列
    """
    vif_df = pd.DataFrame()
    vif_df["feature"] = dataframe.columns
    vif_df["VIF Factor"] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
    high_vif_cols = list(vif_df[vif_df["VIF Factor"] >= thresh]["feature"])
    print("High VIF Columns: ", high_vif_cols)
    low_vif_cols = list(vif_df[vif_df["VIF Factor"] < thresh]["feature"])
    return low_vif_cols

columns = get_low_vif_cols(all_data[columns])
```

#### 3.3.3 其他处理

对于 xgboost 模型，我们尝试了 PCA 进行多重共线性的处理：

```python

# 数据标准化
scaler = StandardScaler()
all_data[columns] = scaler.fit_transform(all_data[columns])
pca = PCA(n_components=10,copy=True,random_state=42)
all_data[columns] = pca.fit_transform(all_data[columns])
```

### 3.4 模型选择

在星级预测中，我们尝试了逻辑回归、决策树、随机森林和 xgboost 模型，最终发现 xgboost 的综合表现最佳。因此在之后的信用预测中，我们仅使用了 xgboost 模型。

此外，我们还尝试使用 optuna 对 xgboost 模型进行自动调参，相关代码如下：

```python
def objective(trial):
    # 定义参数空间
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': trial.suggest_float('eta', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
        'lambda': trial.suggest_float('lambda', 0, 1),
        'alpha': trial.suggest_float('alpha', 0, 1),
    }

    # 定义XGBoost模型
    model = XGBClassifier(random_state=42, **params)

    # 训练模型
    model.fit(X_train, y_train)

    # 验证模型
    y_pred = model.predict(X_valid)
    y_pred = xgb_le.inverse_transform(y_pred)

    # 计算评价指标，这里使用准确率作为评价指标
    accuracy = accuracy_score(y_valid, y_pred)

    # 保存最佳模型
    if trial.should_prune():  # 检查是否需要提前停止训练
        raise optuna.exceptions.TrialPruned()
    else:
        dump(model, xgb_name)  # 保存模型

    # 返回评价指标的负值，因为Optuna默认最小化目标函数
    return -accuracy


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print('Best parameters:', study.best_params)
print('Best objective value:', study.best_value)

```

### 3.5 模型评估

包括准确率、混淆矩阵、精确率、召回率、F1分数和 Cohen's Kappa 系数。代码如下：

```python

# 5 模型评估
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print('准确率为：', accuracy)
# 计算混淆矩阵并保存为图片
save_path = PREFIX + 'confusion_matrix_xgb.png'
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot()
disp.figure_.savefig(save_path)
precision = precision_score(y_test, y_pred, average='macro')  # 计算宏平均精确率
recall = recall_score(y_test, y_pred, average='macro')  # 计算宏平均召回率

print("精确率为: ", precision)
print("召回率为: ", recall)
f1 = f1_score(y_test, y_pred, average='macro')  # 计算宏平均F1分数
print("F1分数为: ", f1)
# 计算Cohen's Kappa系数
kappa = cohen_kappa_score(y_test, y_pred)
print("Cohen's Kappa系数为: ", kappa)

print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

```

具体代码输出见星级预测和信用预测的模型评估部分。

### 3.6 模型应用

代码如下：

```python
#  读取待预测数据
result = pd.read_csv(predict_name)
# 预测
y_pred = model.predict(result[columns])
# 如果使用 xgboost 模型，需要将预测结果反编码
y_pred = xgb_le.inverse_transform(y_pred)
# 保存预测结果至 result.csv 文件
result['star_level'] = y_pred
result = result.loc[:, ['uid', 'star_level']]
result.to_csv(PREFIX + 'result.csv', index=False)

```

## 4. 星级预测

### 4.1 数据收集

首先，pri_star_info 的不重复 uid 共有290658个。而 pri_cust_asset_info 中的交集有290642个，基本全覆盖，因此我们主要使用 pri_cust_asset_info 的数据。

在字段选择方面，我们选择了 pri_cust_base_info （基本信息） 中的部分字段以及 pri_cust_asset_info （存款汇总信息） 中的所有字段。考虑到一个 uid 可能会有多个存款账号和对应的存款汇总信息，我们使用 SUM 的方法对 uid 去重来保证唯一。具体的 SQL 语句如下：

```sql
-- 用于训练和模型应用的集合
select star_info.uid as uid,
       star_level,
       sex,
       marrige,
       education,
       is_black,
       is_contact,
       all_bal,
       avg_mth,
       avg_qur,
       avg_year,
       sa_bal,
       td_bal,
       fin_bal,
       sa_crd_bal,
       td_crd_bal,
       sa_td_bal,
       ntc_bal,
       td_3m_bal,
       td_6m_bal,
       td_1y_bal,
       td_2y_bal,
       td_3y_bal,
       td_5y_bal,
       oth_td_bal,
       cd_bal
from pri_star_info as star_info
         left join (
    select uid,
           SUM(all_bal)    AS all_bal,
           SUM(avg_mth)    AS avg_mth,
           SUM(avg_qur)    AS avg_qur,
           SUM(avg_year)   AS avg_year,
           SUM(sa_bal)     AS sa_bal,
           SUM(td_bal)     AS td_bal,
           SUM(fin_bal)    AS fin_bal,
           SUM(sa_crd_bal) AS sa_crd_bal,
           SUM(td_crd_bal) AS td_crd_bal,
           SUM(sa_td_bal)  AS sa_td_bal,
           SUM(ntc_bal)    AS ntc_bal,
           SUM(td_3m_bal)  AS td_3m_bal,
           SUM(td_6m_bal)  AS td_6m_bal,
           SUM(td_1y_bal)  AS td_1y_bal,
           SUM(td_2y_bal)  AS td_2y_bal,
           SUM(td_3y_bal)  AS td_3y_bal,
           SUM(td_5y_bal)  AS td_5y_bal,
           SUM(oth_td_bal) AS oth_td_bal,
           SUM(cd_bal)     AS cd_bal
    from pri_cust_asset_info
    group by uid
    ) as asset on star_info.uid = asset.uid
         left join (
    select uid,
           sex,
           marrige,
           education,
           is_black,
           is_contact
    from pri_cust_base_info
    ) as base_info on star_info.uid = base_info.uid;
```

数据盘点如下：

![](https://cdn.hcplantern.cn/img/2023/05/21/20230521-135551.png)



### 4.2 数据处理与模型训练

在使用 k 值编码、处理极端值、标准化、处理相关性和多重共线性时，各模型的准确率如下：
- xgboost：0.8182927704788571
- 逻辑回归：0.7679948309282791
- 决策树： 0.7419771699332328
- 随机森林：0.7587622944935027

可以看到 xgboost 准确率最高。

而使用独热编码、**不处理极端值**、**不标准化**、**不处理相关性和多重共线性**时，各模型的准确率如下：
- xgboost：0.9163615478498097
- 逻辑回归：无
- 决策树： 0.8928422715198506
- 随机森林：0.9090099791801278

可以看到准确率有很大进步。接下来将仅使用 xgboost 进行进一步优化。

首先，我们可以找出哪些步骤可能会降低准确率。
- 在第一次处理的基础上去除极端值处理和标准化步骤，准确率如下：0.8179338071649077，区别不大；
- 在此基础上再去除相关性和多重共线性的处理，准确率上升至 0.9159020748079546。可以得出结论是相关性和多重共线性的处理降低了准确率。
- 再将 k值编码替换为独热编码，准确率进一步上升至 0.9163615478498097。

以上探索和xgboost模型的特性相吻合。xgboost是基于决策树的算法，不需要去除极端值和标准化，也对多重共线性具有一定的鲁棒性。此外，xgboost 需要独热编码。因此，我们最终的处理流程为：独热编码、**不处理极端值**、**不标准化**、**不处理相关性和多重共线性**。

随后，我们使用 optuna 自动调参，迭代 50 次之后在测试集上的最高准确率为 0.9181779022183932，参数为：`Best parameters: {'eta': 0.04748675169233668, 'max_depth': 10, 'subsample': 0.8378975895936535, 'colsample_bytree': 0.9319495724301979, 'lambda': 0.6795430340745882, 'alpha': 0.7546130905762063}`。故使用该模型。


### 4.3 模型评估

模型评估如下：

```text
准确率为： 0.9171440878742193
精确率为:  0.6230719278808886
召回率为:  0.6516028293718341
F1分数为:  0.6356226262229501
Cohen's Kappa系数为:  0.867729858093816
              precision    recall  f1-score   support

           1       0.98      0.97      0.98     25822
           2       0.89      0.91      0.90      8833
           3       0.89      0.90      0.89      7754
           4       0.59      0.54      0.56      1981
           5       0.62      0.67      0.64      1353
           6       0.71      0.82      0.76       669
           7       0.33      0.31      0.32        13
           8       0.60      0.75      0.67         4
           9       0.00      0.00      0.00         1

    accuracy                           0.92     46430
   macro avg       0.62      0.65      0.64     46430
weighted avg       0.92      0.92      0.92     46430

```

混淆矩阵如下：

![](https://cdn.hcplantern.cn/img/2023/05/21/20230521-135809.png)

### 4.4 模型预测

预测结果在 `result/star_result.csv` 中。


## 5. 信用预测

### 5.1 数据收集

`pri_credit_info` 表中有38309个uid。
以下是和贷款有关的表以及其和 `pri_credit_info` 的 uid 交集数量：
- `pri_cust_liab_info` 即贷款账号汇总信息表。表中由于记录了时间轴，所以选择最新的记录。剩下16662条，覆盖率 43.5%，可见覆盖率仍然不高，有大量缺失值，从而导致信用预测较为困难。
- `pri_cust_liab_acct_info` 贷款账号信息，覆盖率 0。
- `pri_cust_base_info` 基本信息，选择的字段和星级预测一样，覆盖率100%。
- `dm_v_tr_contract_mx` 贷款合同记录，覆盖率 73.5%，但是比较难决策使用哪些字段，故未选择。
- `dm_v_as_djkfq_info` 信用卡分期信息，交集 uid 数量 233，覆盖率0.6%，未选择。
- `dm_v_as_djk_info` 信用卡开户信息，交集 uid 数量 6379，16.7%。未选择。
- `dm_v_tr_djk_mx` 信用卡交易纪录，覆盖率 10.4%，未选择。

综上，我们初步选择 `pri_credit_info`，`pri_cust_base_info` 和 `pri_cust_liab_info`（每个人的最新数据）

最终我们还选择了哪些表的数据：
- 活期交易的数据 dm_v_tr_sa_mx （覆盖 38309 100%）
- 存款数据 pri_cust_asset_info （覆盖 14957 39.0%）
- 加入贷款还本数据 dm_v_tr_huanb_mx （覆盖 15327 40.0%）


具体的SQL语句为：

```sql
-- 选择 pri_credit_info pri_cust_base_info 和 pri_cust_liab_info（每个人的最新数据）再加入加入活期交易的数据 dm_v_tr_sa_mx
-- 再加入存款信息试试？
-- 再加入贷款还本试试？
-- 再加入贷款还息试试？
select credit_info.uid as uid,
       credit_level,
       sex,
       marrige,
       education,
       is_black,
       is_contact,
       all_bal,
       bad_bal,
       due_intr,
       norm_bal,
       delay_bal,
       tran_amt,
       all_bal_from_asset,
       avg_mth,
       avg_qur,
       avg_year,
       sa_bal,
       td_bal,
       fin_bal,
       sa_crd_bal,
       td_crd_bal,
       sa_td_bal,
       ntc_bal,
       td_3m_bal,
       td_6m_bal,
       td_1y_bal,
       td_2y_bal,
       td_3y_bal,
       td_5y_bal,
       oth_td_bal,
       cd_bal,
       huanb_tran_amt,
       huanx_tran_amt
from dm.pri_credit_info as credit_info
         left join (
--  基本信息
    select * from pri_cust_base_info
    ) as base_info on base_info.uid = credit_info.uid
         left join (
--  贷款账户汇总中的最新数据
    select *
    from pri_cust_liab_info
             right join(
        select uid, max(etl_dt) as etl_dt
        from pri_cust_liab_info
        group by uid
        ) as max_etl_dt using (uid, etl_dt)
    ) as liab_info on liab_info.uid = credit_info.uid
         left join (
--  活期交易中的总交易金额
    select uid, sum(tran_amt) as tran_amt
    from pri_credit_info
             left join stream.dm_v_tr_sa_mx using uid
    group by uid
    ) as sa_mx on sa_mx.uid = credit_info.uid
         left join (
--  存款信息
    select uid,
           SUM(all_bal)    AS all_bal_from_asset,
           SUM(avg_mth)    AS avg_mth,
           SUM(avg_qur)    AS avg_qur,
           SUM(avg_year)   AS avg_year,
           SUM(sa_bal)     AS sa_bal,
           SUM(td_bal)     AS td_bal,
           SUM(fin_bal)    AS fin_bal,
           SUM(sa_crd_bal) AS sa_crd_bal,
           SUM(td_crd_bal) AS td_crd_bal,
           SUM(sa_td_bal)  AS sa_td_bal,
           SUM(ntc_bal)    AS ntc_bal,
           SUM(td_3m_bal)  AS td_3m_bal,
           SUM(td_6m_bal)  AS td_6m_bal,
           SUM(td_1y_bal)  AS td_1y_bal,
           SUM(td_2y_bal)  AS td_2y_bal,
           SUM(td_3y_bal)  AS td_3y_bal,
           SUM(td_5y_bal)  AS td_5y_bal,
           SUM(oth_td_bal) AS oth_td_bal,
           SUM(cd_bal)     AS cd_bal
    from pri_cust_asset_info
    group by uid
    ) as asset_info on asset_info.uid = credit_info.uid
         left join (
--  贷款还本
    select uid, sum(tran_amt) as huanb_tran_amt from stream.dm_v_tr_huanb_mx group by uid
    ) as huanb_mx on huanb_mx.uid = credit_info.uid
         left join (
--  贷款还息
    select uid, sum(tran_amt) as huanx_tran_amt from stream.dm_v_tr_huanx_mx group by uid
    ) as huanx_mx on huanx_mx.uid = credit_info.uid
;

```

数据盘点如下：

![](https://cdn.hcplantern.cn/img/2023/05/21/20230521-135845.png)


### 5.2 数据处理与模型训练

数据处理与星级预测保持一致，即独热编码、不处理极端值、不标准化、不处理相关性和多重共线性。

初步选择的数据所得到的预测结果为：
- 准确率为： 0.6171119488608425
- 精确率为:  0.25872438908525286
- 召回率为:  0.20992359762327606
- F1分数为:  0.18087108381346048
- Cohen's Kappa系数为:  0.04565054555257908

Kappa 系数过低。

考虑到贷款信息的缺失率过高，训练结果十分不乐观。因此，我们考虑加入其他交易相关和存款相关的表。

加入活期交易的数据后，结果为：
- 准确率为： 0.6371086707097198
- 精确率为:  0.2827725417353988
- 召回率为:  0.24819520524702265
- F1分数为:  0.2408536372860833
- Cohen's Kappa系数为:  0.2057259691575546

结果有明显提升。再次改进，加入存款数据后，结果为：
- 准确率为： 0.6446484182920833
- 精确率为:  0.29658017545116305
- 召回率为:  0.2717755154273365
- F1分数为:  0.27437819309309025
- Cohen's Kappa系数为:  0.26415719202550736

进一步提升。再次改进，加入贷款还本数据后，结果为：
- 准确率为： 0.6510408129814784
- 精确率为:  0.3020794591066001
- 召回率为:  0.27666246290123775
- F1分数为:  0.27950438370294706
- Cohen's Kappa系数为:  0.28058935379854233

进一步提升。最后，我们考虑加入贷款还息数据，但结果准确率下降。因此以上就是我们所选择的数据字段。

此外，和星级预测一样，我们使用自动调参得到的参数为：

`Best parameters: {'eta': 0.03964175443917428, 'max_depth': 8, 'subsample': 0.6568986608698768, 'colsample_bytree': 0.7628650444384782, 'lambda': 0.8002260883099892, 'alpha': 0.5740743731896754}`

该参数训练出的模型在测试集上的结果为：
- 准确率为： 0.660547451237502
- 精确率为:  0.3005560558966721
- 召回率为:  0.26820500552844023
- F1分数为:  0.26374363764366254
- Cohen's Kappa系数为:  0.2797740020117033

除了准确率其他结果都有所下降，因此不采用自动调参后的模型。


### 5.3 模型评估

```text
准确率为： 0.6520527739080554
精确率为:  0.3040855372140216
召回率为:  0.27854814759050317
F1分数为:  0.281890874628837
Cohen's Kappa系数为:  0.2850402813512123
              precision    recall  f1-score   support

          35       0.00      0.00      0.00         3
          50       0.00      0.00      0.00        50
          60       0.73      0.86      0.79      7520
          70       0.46      0.39      0.42      3230
          85       0.33      0.14      0.19      1400

    accuracy                           0.65     12203
   macro avg       0.30      0.28      0.28     12203
weighted avg       0.61      0.65      0.62     12203

```

混淆矩阵如下：

![](https://cdn.hcplantern.cn/img/2023/05/21/20230521-135917.png)

### 5.4 模型应用

预测结果在 `result/credit_resit.csv` 中