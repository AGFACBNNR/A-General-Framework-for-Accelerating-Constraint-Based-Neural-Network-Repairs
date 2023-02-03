import json
import seaborn as sns
import matplotlib.pyplot as plt

PathRoot='MM'
modelNameDict=['cifar10','fmnist','mnist']
abDict=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
marker = ['^','3','P','x','_','p','|','*','o','d']

def draw(data,ind,type,baseData):
    for i,line in enumerate(data):
        plt.plot(abDict,line,label=rf'$\beta$={abDict[i+1]}')
    if len(baseData) > 0:
        plt.plot(abDict,baseData,label=f'original')
    plt.legend(fontsize=10)
    if type=="time":
        plt.ylabel("time(s)",fontsize=20)
    elif type == "improve":
        plt.ylabel("improvement(%)",fontsize=20)
    else:
        plt.ylabel("drawdown(%)",fontsize=20)
    plt.xlabel(r"$\alpha$",fontsize=20)
    plt.xticks(fontsize=20,rotation=90)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{modelNameDict[ind]}_{type}.png', dpi=300)
    plt.clf()

for ind in [0,1,2]:
    TimeLineB=[]
    DrawdownLineB=[]
    ImproveLineB=[]
    TimeInfo=[]
    DrawdownInfo=[]
    ImproveInfo=[]
    fileName = f'{PathRoot}{modelNameDict[ind]}_1.0_1.0_time.txt'
    try :
        lines = open(fileName).readlines()
        assert (len(lines)>1)
        time = float(lines[-2])
        info = lines[-1].split(",")
        assert (len(info)==4)
        for j in range(len(abDict)):
                    TimeLineB.append(time)
                    DrawdownLineB.append((float(info[0])-float(info[1]))*100)
                    ImproveLineB.append((float(info[3])-float(info[2]))*100)
    except:
            TimeLineB=[]
            DrawdownLineB=[]
            ImproveLineB=[]

    for a in abDict[1:]:
            TimeLine = [0.0]
            DrawdownLine = [0.0]
            ImproveLine = [0.0]
            flag = 1
            for b in abDict[1:]:
                fileName = f'{PathRoot}{modelNameDict[ind]}_{a}_{b}_time.txt'
                try :
                    lines = open(fileName).readlines()
                    assert (len(lines)>1)
                    time = float(lines[-2])
                    info = lines[-1].split(",")
                    assert (len(info)==4)
                    TimeLine.append(time)
                    DrawdownLine.append((float(info[0])-float(info[1]))*100)
                    ImproveLine.append((float(info[3])-float(info[2]))*100)
                except:
                    flag = 0
                    print(fileName,len(lines),len(info))
                    break
            if (not flag):
                break
            TimeInfo.append(TimeLine)
            DrawdownInfo.append(DrawdownLine)
            ImproveInfo.append(ImproveLine)
    
    draw(TimeInfo,ind,"time",TimeLineB)
    draw(DrawdownInfo,ind,"drawdown",DrawdownLineB)
    draw(ImproveInfo,ind,"improve",ImproveLineB)
        #     TimeInfo.append(TimeLine)
        #     DrawdownInfo.append(DrawdownLine)
        #     ImproveInfo.append(ImproveLine)
        # sns.set()
        # ax = sns.heatmap(TimeInfo,xticklabels=abDict, yticklabels=abDict)
        # plt.savefig(f'{modelNameDict[ind]}_{layer}_Time.png', dpi=300)
        # plt.clf()
        # sns.set()
        # ax = sns.heatmap(DrawdownInfo,xticklabels=abDict, yticklabels=abDict)
        # plt.savefig(f'{modelNameDict[ind]}_{layer}_Drawdown.png', dpi=300)
        # plt.clf()
        # sns.set()
        # ax = sns.heatmap(ImproveInfo,xticklabels=abDict, yticklabels=abDict)
        # ax.get_figure().savefig(f'{modelNameDict[ind]}_{layer}_Improve.png', dpi=300)
        # plt.clf()

