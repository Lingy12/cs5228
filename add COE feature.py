import pandas as pd


# 读取Excel数据
df = pd.read_csv('C:/Users/X/Desktop/NUS/CS5228/project/sg-coe-prices.csv')
df_other = pd.read_csv('C:/Users/X/Desktop/NUS/CS5228/project/data after training/train_with_mrt_mall_school.csv')

# df_other['rent_approval_date'] = pd.to_datetime(df_other['rent_approval_date'])
# 将月份转化为数字
month_to_num = {'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06', 'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'}
df['month'] = df['month'].map(month_to_num)
#
df['year'] = df['year'].astype(str)  # 将年份列转换为字符串
df['month'] = df['month'].astype(str)  # 将月份列转换为字符串

# 使用 str.cat() 方法将年份和月份合并成日期列
df['rent_approval_date'] = df['year'].str.cat(df['month'], sep='-')
# print(df['rent_approval_date'])

#
# 根据类别和时间计算平均值
result = df.groupby(['year', 'category', 'rent_approval_date']).agg({'price': 'mean', 'quota': 'mean', 'bids': 'mean'}).reset_index()
result = result.drop(columns = ['category','year','quota','bids'])
result = result.groupby('rent_approval_date')['price'].mean().reset_index()
# print(result['rent_approval_date'][12])
result_merged = pd.merge(result, df_other, on='rent_approval_date', how='inner')
# print(result_merged)

# result_merged['rent_approval_date'] = pd.to_datetime(result_merged['rent_approval_date'], format='%Y-%m')

result_merged.to_csv('C:/Users/X/Desktop/NUS/CS5228/project/data after training/train_with_mrt_mall_school_coe.csv', index=False)

# train_final = ('C:/Users/X/Desktop/NUS/CS5228/project/data after training/train_with_mrt_mall_school_coe.csv')

