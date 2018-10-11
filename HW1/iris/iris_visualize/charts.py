import numpy as np
import matplotlib.pyplot as plt
def test_key_val(key,val):
    if len(key) is not len(val):
        print('Some Error in your input data!')
        return 0
def test_datas(datas):
    tmp = len(datas[0])
    for i in datas:
        if len(i) is not tmp:
            print('Some Error in your input data!')
            return 0
    return 1
def others(plt,title,filename,figsize=(8,6),dpi=70):
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename)
def bar(keys,values,mean,var,title='',filename='',figsize=None, dpi=None):
    test_key_val(keys,values)

    plt.bar(keys, values,width=0.1,color='aquamarine')  # tick_label = Key
    plt.axvline(x=mean,color='red',linewidth=4)
    plt.axvline(x=mean+var,linewidth=4)
    plt.axvline(x=mean-var,linewidth=4)
    others(plt,title,filename,figsize,dpi)
    plt.show()
def stack_bar(keys,labels,*datas,title='',filename='',figsize=None, dpi=None):
    if not test_datas(datas): return 
    plt.figure() # 定義一個圖像窗口
    plt.subplot() # 創建小圖(分層圖)
    index = np.arange(len(keys)) # 形成等差數列
    for i in range(len(labels)):
        # 每次疊上 data[:i] 的和
        plt.bar(index,datas[i],width=0.7,bottom=np.sum(datas[:i],axis=0),tick_label=keys)
    plt.legend(labels)
    others(plt,title,filename,figsize,dpi)
    plt.show()
def parrallel_bar(keys,labels,*datas,title='',filename='',figsize=None, dpi=None):
    if not test_datas(datas): return 
    lst = []
    for i in datas:
        lst += i
    datas = lst
    x = list(range(len(keys)))
    n = len(labels) #種類個數
    total_width= 0.8 #所有長條的寬度和
    Width = total_width / n  #各個長條的寬度

    for i in range(n): #每次畫一種類的長條圖
        tmp_lst = datas[i*len(x):(i+1)*len(x)]            
        plt.bar(x,tmp_lst,width=Width,label=labels[i],tick_label=keys)
        # width: 該資料長條的寬度
        # label: 該種類標籤名字
        # tick_label: bar的標籤， 這個例子就是 Sun Mon Tue Wed Thu Fri Sat
        # fc(facecolor): 顏色
        for j in range(len(x)):  
            x[j] = x[j] + Width
     
    plt.legend()  # 顯示圖例
    others(plt,title,filename,figsize,dpi)
    plt.show() # 顯示圖像 
def radar(keys,values,title='',filename='',figsize=None, dpi=None):
    test_key_val(keys,values)
    cnt = sum(values)/len(values)
    for i in range(len(values)):
        values[i] /= (cnt/2.5)
    labels = np.array(keys)
    datas = np.array(values)
    angles = np.linspace(0, 2*np.pi, len(keys), endpoint=False)
    datas = np.concatenate((datas, [datas[0]])) # 閉合圓形
    angles = np.concatenate((angles, [angles[0]])) # 閉合圓形

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True) # 轉換極座標
    ax.plot(angles, datas, 'bo-', linewidth=2) # 畫線
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.set_title(title)
    ax.set_rlim(0,5) # 標出距離圓心
    others(plt,title,filename,figsize,dpi)
    plt.show()
def pie(keys,values,title='',filename='',figsize=None, dpi=None):
    test_key_val(keys,values)
    colors = ['lightcoral','palegreen','aquamarine']
    plt.pie(x = values, # 繪圖數據
            labels=keys,
            autopct='%.1f%%', # 設置百分比的格式，這裏保留一位小數
            pctdistance=0.8,  # 設置百分比標籤與圓心的距離
            textprops={'fontsize':12, 'color':'k'}, # 設置文本標籤的屬性值
            colors=colors
            )
    others(plt,title,filename,figsize,dpi)
    plt.show()
def line(keys,labels,*datas,title='',filename='',figsize=None, dpi=None):
    if not test_datas(datas): return 
    plt.figure() # 定義一個圖像窗口
    plt.subplot() # 創建小圖(分層圖)
    index = np.arange(len(keys)) # 形成等差數列
    for i in datas:
        x = [p[0] for p in i]
        y = [p[1] for p in i]
        plt.plot(x,y,'-o')
    plt.legend(labels)
    others(plt,title,filename,figsize,dpi)
    plt.show()
