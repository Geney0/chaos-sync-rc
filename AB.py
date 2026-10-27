# # CTR点击率=点击用户数/浏览用户数（去重后独立访客数
# p-value < 0.05
#
# import pandas as pd
#
# data = pd.read_csv('ab_data.csv')
# print(data)
#
# # 了解数据整体情况
# size=len(data)
# print("数据数量：",size)
# user_size=len(data['id'].drop_duplicates(inplace=True))
# print("用户人数：",user_size)
# days = pd.to_datetime(data['timestamp'].tail(1).values)-pd.to_datetime(data['timestamp'].head(1).values)
# print("测试时长:",days)









def add_print(a, b):
    print(a + b)          # 只打印，不交出去

def add_return(a, b):
    return a + b          # 交出去，不打印

x = add_print(1, 2)       # 屏幕上会显示 3
y = add_return(1, 2)      # 屏幕上什么都没显示

print("x 是：", x)
print("y 是：", y)