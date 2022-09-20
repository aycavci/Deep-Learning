import os
import matplotlib.pyplot as plt
import numpy as np

from numpy import average

directory = "./output/cityscapes/logToGraph/"

def folderToSummary(folder):
    epochLosses = [0]*700
    validationLosses = [0]*700
    validationMeanUIs = [0]*700

    # Harvest data from all log files in the directory
    for filename in os.listdir(folder):
        fullFileName = os.path.join(folder, filename)
        if ".log" in filename:
            with open(fullFileName) as file:
                lines = file.readlines()
            for index, line in enumerate(lines):
                #2022-03-27 16:24:33,715 Epoch: [21/484] Iter:[0/185], Time: 4.69, lr: [0.00960864318920482], Loss: 0.334865
                if "Epoch: [" in line:
                    if "/484]" in line:
                        Epoch = int(line.split("Epoch: [",1)[1].split("/484] Iter:",1)[0])
                    elif "/653]" in line:
                        Epoch = int(line.split("Epoch: [",1)[1].split("/653] Iter:",1)[0])
                    IterText = line.split("Iter:[",1)[1].split("], Time",1)[0].split("/",1)
                    Iteration = int(IterText[0])
                    Iterations = int(IterText[1])
                    Loss = float(line.split("Loss: ",1)[1].split("\n",1)[0])
                    if Iterations == 185 and Iteration == 100:
                        if epochLosses[Epoch]!=0:
                            # Duplicate
                            print()
                        else:
                            epochLosses[Epoch] = Loss
                # 2022-03-27 15:58:25,453 Loss: 0.283, MeanIU:  0.4567, Best_mIoU:  0.4567
                if "MeanIU" in line and not "Pixel" in line:
                    epochLine = lines[index-7]
                    checkpointLine = lines[index-5]
                    if "Epoch: [" in epochLine:
                        if "/484]" in epochLine:
                            lastEpoch = int(epochLine.split("Epoch: [",1)[1].split("/484] Iter:",1)[0])
                        elif "/653]" in epochLine:
                            lastEpoch = int(epochLine.split("Epoch: [",1)[1].split("/653] Iter:",1)[0])
                        else:
                            print()
                        
                    elif "checkpoint" in checkpointLine:
                        lastEpoch = int(checkpointLine.split("(epoch ",1)[1].split(")",1)[0])
                        print()
                    else:
                        # no corresponding epoch found
                        print()
                    MeanUI = float(line.split("MeanIU: ",1)[1].split(", Best_mIoU",1)[0])
                    valLoss = float(line.split("Loss: ",1)[1].split(", MeanIU",1)[0])
                    validationLosses[lastEpoch-1] = valLoss
                    validationMeanUIs[lastEpoch-1] = MeanUI

    # prepare the data for plotting by removing all empty entrys
    trainingSummary = []
    validatingSummary = []
    for index, valLoss in enumerate(validationLosses):
        if valLoss != 0:#and averageTrainLosses[index] != 0:
            epoch = index+1
            singleSummary = [epoch, valLoss, validationMeanUIs[index]]
            validatingSummary.append(singleSummary.copy())
        if epochLosses[index]!=0:
            lossElement = [index+1,epochLosses[index]]
            trainingSummary.append(lossElement.copy())

    validatingSummary = np.array(validatingSummary)
    trainingSummary = np.array(trainingSummary)
    return validatingSummary, trainingSummary

modelSummaries = []
subDirectories = [folder[0] for folder in os.walk(directory)]
for subDirectory in subDirectories:
    if subDirectory != directory:
        parts = subDirectory.split(directory,1)
        modelName = parts[len(parts)-1]
        validatingSummary, trainingSummary = folderToSummary(subDirectory)
        modelSummaries.append([modelName, validatingSummary,trainingSummary])

# plot loss graph
fontSize = 20
plt.figure(figsize=(15,6))
for model in modelSummaries:
    name = model[0]
    validatingSummary = model[1]
    trainingSummary = model[2]

    epochs = validatingSummary[:,0].astype(int)
    validationLoss = validatingSummary[:,1]
    validationMeanIU = validatingSummary[:,2]
    epochsTraining = trainingSummary[:,0].astype(int)
    trainingLoss =  trainingSummary[:,1]

    plt.plot(epochsTraining, trainingLoss, label = name + " training loss")
    plt.plot(epochs, validationLoss, label = name + " validation loss")

plt.xlim([1, max(epochs)])
plt.ylim([0, 1])
#plt.xticks(epochs,epochs)
plt.xticks(fontsize = fontSize, rotation = 0)
plt.yticks(fontsize = fontSize)
plt.xlabel('Epoch', fontsize = fontSize)
plt.ylabel('Loss', fontsize = fontSize)
plt.legend(fontsize = fontSize)
plt.savefig("./resultsLoss.png", dpi = 300, bbox_inches='tight') # when saving, specify the DPI

# plot meanIoU graph
plt.figure(figsize=(15,6))
for model in modelSummaries:
    name = model[0]
    validatingSummary = model[1]
    trainingSummary = model[2]

    epochs = validatingSummary[:,0].astype(int)
    validationLoss = validatingSummary[:,1]
    validationMeanIU = validatingSummary[:,2]
    epochsTraining = trainingSummary[:,0].astype(int)
    trainingLoss =  trainingSummary[:,1]

    plt.plot(epochs, validationMeanIU, label = name + " validation mean IoU")
plt.xlim([1, max(epochs)])
plt.ylim([0, 1])
plt.xticks(fontsize=14, rotation=90)
#plt.xticks(epochs,epochs)
plt.xticks(fontsize = fontSize, rotation = 0)
plt.yticks(fontsize = fontSize)
plt.xlabel('Epoch', fontsize = fontSize)
plt.ylabel('Mean IoU', fontsize = fontSize)
plt.legend(fontsize = fontSize)
plt.savefig("./resultsMeanIoU.png", dpi = 300, bbox_inches='tight') # when saving, specify the DPI