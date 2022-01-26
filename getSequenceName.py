import os

def getSequenceName(thispath : str, dataRoot : str = '/n/fs/sun3d/data/') -> str:
    sequenceName = thispath[len(dataRoot):]
    while sequenceName[0]=='/':
        sequenceName = sequenceName[1:]
    while sequenceName[-1]=='/':
        sequenceName =sequenceName[:-1]

    return sequenceName
