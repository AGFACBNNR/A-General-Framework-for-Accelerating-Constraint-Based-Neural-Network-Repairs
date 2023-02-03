import json
import seaborn as sns
import matplotlib.pyplot as plt

PathRoot='MMRes'
modelNameDict=['cifar10','fmnist','mnist']
layerDict=[12,7,6]
abDict=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]


def draw(data,ind,type,baseData):
    for i,line in enumerate(data):
        plt.plot(abDict[:len(line)],line,label=rf"$\beta$={abDict[i+1]}")
    if len(baseData) > 0:
        plt.plot(abDict,baseData,label=f'original')
    plt.legend(fontsize=10)
    if type=="time":
        plt.ylabel("time(s)",fontsize=20)
    elif type == "improve":
        plt.ylabel("improvement(%)",fontsize=20)
    else:
        plt.ylabel("drawdown(%)",fontsize=20)
    plt.xlabel(r'$\alpha$',fontsize=20)
    plt.xticks(fontsize=20,rotation=90)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{modelNameDict[ind]}_{layer}_{type}.png', dpi=300)
    plt.clf()

for ind in [0,1,2]:
    layer = layerDict[ind]
    TimeLineB=[]
    DrawdownLineB=[]
    ImproveLineB=[]
    TimeInfo=[]
    DrawdownInfo=[]
    ImproveInfo=[]
    fileName = f'{PathRoot}_{modelNameDict[ind]}_1.0_1.0_{layer}.json'
    try :
        info = json.load(open(fileName))
        for j in range(len(abDict)):
            TimeLineB.append(info['Time']) 
            DrawdownLineB.append(info['before']['test_identity']-info['after']['test_identity'])
            ImproveLineB.append(info['after']['test_corrupted']-info['before']['test_corrupted'])
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
                fileName = f'{PathRoot}_{modelNameDict[ind]}_{a}_{b}_{layer}.json'
                try :
                    info = json.load(open(fileName))
                    TimeLine.append(info['Time'])
                    DrawdownLine.append(info['before']['test_identity']-info['after']['test_identity'])
                    ImproveLine.append(info['after']['test_corrupted']-info['before']['test_corrupted'])
                except:
                    flag = 0
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

