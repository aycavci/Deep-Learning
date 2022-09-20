import os.path
from os import path

with open('test.lst') as f:
    lines = f.readlines()

with open('newTest.lst', 'w') as f:
    for index, line in enumerate(lines):
        fileText = line.split("test/",1)[1] 
        fileText = fileText.split("leftImg8bit",1)[0] 
        fileText += "gtFine_labelIds.png"
        fileText = "gtFine/test/"+fileText
        newLine = line.replace('\n', '\t')
        newLine += fileText
        f.write(newLine)
        f.write('\n')




    #gtFine/test/berlin/berlin_000000_000019_gtFine_labelIds.png

    