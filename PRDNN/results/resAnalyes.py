import json
import seaborn as sns
import matplotlib.pyplot as plt

PathRoot='MMRes'
modelNameDict=['cifar10','fmnist','mnist']
layerDict=[(12,14),(7,9),(6,8)]
abDict=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for ind in [0,1,2]:
    for layer in layerDict[ind]:
        TimeLineB=[]
        DrawdownLineB=[]
        ImproveLineB=[]
        fileName = f'{PathRoot}_{modelNameDict[ind]}_1.0_1.0_{layer}.json'
        try :
                    info = json.load(open(fileName))
                    for j in range(len(abDict)):
                        TimeLineB.append(info['Time']) 
                        DrawdownLineB.append(info['before']['test_identity']-info['after']['test_identity'])
                        ImproveLineB.append(info['after']['test_corrupted']-info['before']['test_corrupted'])
        except:
                for j in range(len(abDict)):
                    TimeLineB.append(0)
                    DrawdownLineB.append(0)
                    ImproveLineB.append(0)

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
            if (not flag):
                break
            plt.plot(abDict,TimeLine)
            plt.plot(abDict,TimeLineB)
            plt.savefig(f'{modelNameDict[ind]}_{layer}_{a}_Time.png', dpi=300)
            plt.clf()
            plt.plot(abDict,DrawdownLine)
            plt.plot(abDict,DrawdownLineB)
            plt.savefig(f'{modelNameDict[ind]}_{layer}_{a}_Drawdown.png', dpi=300)
            plt.clf()
            plt.plot(abDict,ImproveLine)
            plt.plot(abDict,ImproveLineB)
            plt.savefig(f'{modelNameDict[ind]}_{layer}_{a}_Improv.png', dpi=300)
            plt.clf()
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

