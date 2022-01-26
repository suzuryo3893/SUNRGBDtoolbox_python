import os
import re
import glob
import numpy as np
import numpy.typing as npt
import json
from typing import Sequence,Dict,Any,List

from getSequenceName import getSequenceName
from mBB.projectStructBbsTo2d import projectStructBbsTo2d

def readframeSUNRGBD(thispath : str, dataRoot : str = '/n/fs/sun3d/data/', cls : Sequence[str] = [], bbmode : str ='2Dbb') -> Dict:
    """
    % example code to read annotation from ".json" file.
    % thispath: full path to the data folder.
    % dataRoot: root directory of all data folder.
    % cls : object category of ground truth to load. If not speficfy, the code will load all ground truth. 
    """

    sequenceName  = getSequenceName(thispath, dataRoot)
    if not os.path.exists(thispath):
        return {'sequenceName' : sequenceName, 'valid' : 0}
    
    indd = re.findall('/',sequenceName)
    sensorType = sequenceName[indd[0]+1 : indd[1]]
    # % get K
    with open(os.path.join(thispath, 'intrinsics.txt'),'r') as fID:
        # K = reshape(fscanf(fID,'%f'),[3,3])'
        lines = [re.sub("^\\s*(.+)\\s*$","\\1",x) for x in fID.readlines()]
        _k = [float(y) for y in re.split('\\s+',x) for x in lines]
        K = np.array(_k, dtype=np.float64).reshape((3,3))
    
    #  % get image and depth path
    # depthpath = dir([thispath '/depth/' '/*.png']);
    # depthname = depthpath(1).name;
    # depthpath = [thispath '/depth/' depthpath(1).name];
    depthpath = glob.glob(os.path.join(thispath,'depth','*.png'))
    depthname = os.path.basename(depthpath[0])
    depthpath = os.path.join(thispath,'depth', depthname)
    
    # rgbpath = dir([thispath '/image/' '/*.jpg']);
    # rgbname = rgbpath(1).name;
    # rgbpath = [thispath '/image/' rgbpath(1).name];
    rgbpath = glob.glob(os.path.join(thispath,'image','*.jpg'))
    rgbname = os.path.basename(rgbpath[0])
    rgbpath = os.path.join(thispath,'image', rgbname)
     
    if os.path.exists(os.path.join(thispath,'annotation3Dfinal/index.json')):
        with open(os.path.join(thispath,'annotation3Dfinal/index.json'),"r") as fID:
            annoteImage = json.load(fID)
        # % get Box
        # filename = dir([fullfile(thispath,'extrinsics') '/*.txt']);
        # Rtilt = dlmread([fullfile(thispath,'extrinsics') '/' filename(end).name]);
        # Rtilt = Rtilt(1:3,1:3);
        # anno_extrinsics = Rtilt;
        filename = glob.glob(os.path.join(thispath,'extrinsics','*.txt'))
        Rtilt = np.loadtxt(os.path.join(thispath,'extrinsics',os.path.basename(filename[-1])),delimiter=' ',dtype=np.float64)
        Rtilt = Rtilt[0:3, 0:3]
        anno_extrinsics = Rtilt.copy()
        # % convert it into matlab coordinate
        Rtilt = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) @ Rtilt @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        cnt = 0
        groundtruth3DBB : List[Any] = []
        for obji in range(len(annoteImage['objects'])):
                annoteobject = annoteImage['objects'][obji]
                if annoteobject is not None and annoteobject[0] is not None and 'polygon' in annoteobject[0]:
                    annoteobject = annoteobject[0]
                    box = annoteobject['polygon'][0]

                    # % class name and label 
                    ind = annoteobject['name'].find(':')
                    if ind == -1:
                        classname  = annoteobject['name']
                        labelname = ''
                    else:
                        #if ismember(annoteobject.name(ind-1),{'_',' '}),
                        if annoteobject['name'][ind-1] in ['_',' ']:
                            clname = annoteobject['name'][:ind-1]
                        else:
                            clname = annoteobject['name'][:ind]
                        
                        # %[~,classId]= ismember(clname,classNames);
                        # classname  = clname;
                        # labelname = annoteobject.name(ind+2:end);
                        # %[~,label]= ismember(Labelname,labelNames);
                        classname  = clname
                        labelname = annoteobject['name'][ind+2:]
                    
                    if classname in ['wall','floor','ceiling'] or \
                        len(cls)>0 and not classname in cls:
                        continue

                    # x =box.X;
                    # y =box.Z;
                    # vector1 =[x(2)-x(1),y(2)-y(1),0];
                    # coeff1 =norm(vector1);
                    # vector1 =vector1/norm(vector1);
                    # vector2 =[x(3)-x(2),y(3)-y(2),0];
                    # coeff2 = norm(vector2);
                    # vector2 =vector2/norm(vector2);
                    # up = cross(vector1,vector2);
                    # vector1 = vector1*up(3)/up(3);
                    # vector2 = vector2*up(3)/up(3);
                    # zmax =-box.Ymax;
                    # zmin =-box.Ymin;
                    # centroid2D = [0.5*(x(1)+x(3)); 0.5*(y(1)+y(3))];
                    x = box['X']
                    y = box['Z']
                    vector1 = np.array([x[1]-x[0], y[1]-y[0], 0])
                    coeff1 = np.linalg.norm(vector1)
                    vector1 = vector1/np.linalg.norm(vector1)
                    vector2 = np.array([x[2]-x[1], y[2]-y[1], 0])
                    coeff2 = np.linalg.norm(vector2)
                    vector2 = vector2/np.linalg.norm(vector2)
                    up = np.cross(vector1, vector2)
                    vector1 = vector1*up[2]/up[2] #original bug?
                    vector2 = vector2*up[2]/up[2]
                    zmax = -box['Ymax']
                    zmin = -box['Ymin']
                    centroid2D = np.array([0.5*(x[0]+x[2]), 0.5*(y[0]+y[2])])

                    thisbb : Dict[str,Any] = {}
                    thisbb['basis'] = np.array([vector1, vector2, [0, 0, 1]]); #% one row is one basis
                    thisbb['coeffs'] = np.abs(np.array([coeff1, coeff2, zmax-zmin]))/2
                    thisbb['centroid'] = np.array([centroid2D[0], centroid2D[1], 0.5*(zmin+zmax)])
                    thisbb['classname'] = classname
                    thisbb['labelname'] = labelname
                    thisbb['sequenceName'] = sequenceName
                    #orientation = [([0.5*(x(2)+x(1)),0.5*(y(2)+y(1))] - centroid2D(:)'), 0];
                    orientation = np.append(np.array([0.5*(x[1]+x[0]), 0.5*(y[1]+y[0])]) - centroid2D, 0)
                    thisbb['orientation'] = orientation/np.linalg.norm(orientation)

                    if bbmode == '2Dbb':
                        bb2d,bb2dDraw = projectStructBbsTo2d(thisbb,Rtilt,[],K)
                        # %gtBb2D = crop2DBB(gtBb2D,427,561);
                        thisbb['gtBb2D'] = bb2d[0:4]
                    
                    groundtruth3DBB.append(thisbb)
                    cnt=cnt+1

        if cnt==0:
            groundtruth3DBB = []
    else:
        groundtruth3DBB = []

        filename = glob.glob(os.path.join(thispath,'extrinsics','*.txt'))
        Rtilt = np.loadtxt(os.path.join(thispath,'extrinsics',os.path.basename(filename[-1])),delimiter=' ',dtype=np.float64)
        Rtilt = Rtilt[0:3, 0:3]
        anno_extrinsics = Rtilt.copy()
        # % convert it into matlab coordinate
        Rtilt = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) @ Rtilt @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # % read in room 
    if os.path.exist(os.path.join(thispath, 'annotation3Dlayout/index.json')):
       #jsond=loadjson([thispath '/annotation3Dlayout/index.json'])
       with open(os.path.join(thispath, 'annotation3Dlayout/index.json'),"r") as fID:
           jsond = json.load(fID)
       for objectID in range(jsond['objects']):
            try:
                groundTruth = jsond['objects'][objectID]['polygon'][0]
                numCorners = len(groundTruth['X'])
                gtCorner3D : npt.NDArray = np.zeros((3,2*numCorners))

                # gtCorner3D(1,:) = [groundTruth.X groundTruth.X];
                # gtCorner3D(2,:) = [repmat(groundTruth.Ymin,[1 numCorners]) repmat(groundTruth.Ymax,[1 numCorners])];
                # gtCorner3D(3,:) = [groundTruth.Z groundTruth.Z];
                # gtCorner3D = anno_extrinsics'*gtCorner3D;
                # gtCorner3D = gtCorner3D([1,3,2],:);
                # gtCorner3D(3,:) = -1*gtCorner3D(3,:);
                # gtCorner3D = Rtilt*gtCorner3D;
                gtCorner3D[0,:] = np.hstack((groundTruth['X'], groundTruth['X']))
                gtCorner3D[1,:] = np.hstack((
                    np.tile(groundTruth['Ymin'],(1, numCorners)),
                    np.tile(groundTruth['Ymax'],(1, numCorners)),
                ))
                gtCorner3D[2,:] = np.hstack((groundTruth['Z'], groundTruth['Z']))
                gtCorner3D = anno_extrinsics.T @ gtCorner3D
                gtCorner3D = gtCorner3D[[0,2,1], :]
                gtCorner3D[2,:] = -1 * gtCorner3D[2,:]
                gtCorner3D = Rtilt @ gtCorner3D
            except Exception as e:
                None
            break
       
    data = {
        'sequenceName' : sequenceName,
        'groundtruth3DBB' : groundtruth3DBB,
        'Rtilt' : Rtilt,
        'K' : K,
        'depthpath' : depthpath,
        'rgbpath' : rgbpath,
        'anno_extrinsics' : anno_extrinsics,
        'depthname' : depthname,
        'rgbname' : rgbname,
        'sensorType' : sensorType,
        'valid' : 1,
        'gtCorner3D' : gtCorner3D
    }

    return data
