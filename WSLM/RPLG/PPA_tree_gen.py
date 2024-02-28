'''
此脚本用于产生分割结果的Graph(包含半径等信息)和distance map的数据文件
保存在./graphsTr/下面
'''
import nibabel as nib
from utils import *
from skimage.morphology import skeletonize
import networkx as nx
import warnings
import numpy as np
warnings.filterwarnings("ignore")
join=os.path.join
listdir=os.listdir
from math import sqrt
import pickle
import math
abs=math.fabs
from tqdm import tqdm
#########超参数##############
suffix='_A.nii.gz'#可以指定哪个case来做，.nii.gz为全部
############################

def mask2points(mask,label=1):
    '''
    将骨架表示成节点列表
    '''
    positions = np.argwhere(mask==label)
    points = []
    for position in positions:
        currentPoint =(position[0],position[1],position[2])
        points.append(currentPoint)
    return points

def get_neighbor(mask,loc,label=1):
    #得到位置loc的neighbors
    deepth, height, weight = mask.shape
    I,J,K=loc
    neighbors=[]
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                if min(I+i,J+j,K+k)<0 or I+i>=deepth or J+j>=height or K+k>=weight or (i==j==k==0):
                    continue
                elif mask[I+i,J+j,K+k]==label:
                    neighbors.append((I+i,J+j,K+k))
    return neighbors

def Loc2No(G,loc):
    #由loc得到对应的节点序号
    for k,v in G._node.items():
        if v['loc']==loc:
            return k

def L2(p1,p2):#求p1和p2的L2距离
    assert len(p1)==len(p2)
    d2=0
    for i in range(len(p1)):
        d2+=(p1[i]-p2[i])**2
    return sqrt(d2)

def in_neighbor(q,p,r):
    for i in range(len(q)):
        if abs(p[i]-q[i])>r:
            return False 
    return True

def main():
    all_info={}
    # all_info['Graph']=None
    # all_info['DisMap']=None
    # all_info['KeyPoints']={}#存放末端点，交叉点，最高点最低点等信息
    root='../labelingGT30/'#分割结果的mask路径
    ref_path='../labelingGT30/'
    save_path='./labelingGT30_pre/'
    LargestCC_save_path='./labelingGT30_lcc'
    for path in [save_path,LargestCC_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    showSkel=False
    NN=len(os.listdir(root))
    nn=0
    for file in listdir(root):
        nn+=1
        print('######',nn,'/',NN,file)
        save_name=file.replace('.nii.gz','.pkl')#预处理数据保存名
        if suffix in file and file.replace('.nii.gz','.pkl') not in listdir(save_path) and file in listdir(ref_path):  
            try:
                print('1. get skeleton...')
                nii=nib.load(join(root,file))
                mask=nii.get_fdata()

                #弄成一个label
                mask=(mask>0).astype(np.uint8)

                LargestCC_mask = getLargestCC(mask).astype(int)#获取最大连通分量
                LargestCC_nii = nib.Nifti1Image(LargestCC_mask, nii.affine)
                nib.save(LargestCC_nii, join(LargestCC_save_path,file))

                skeleton = skeletonize(LargestCC_mask)#利用Li的方法或者最大连通骨架
                skel_points=mask2points(skeleton,label=1)
                all_points=mask2points(LargestCC_mask,label=1)#最大连通血管上的点
                
                #显示骨架
                if showSkel:
                    print('We show you the skeleton points...')
                    show_skeleton(skel_points)#显示中心线

                #以最大连通骨架点构建Graph
                print('2.build graph...')
                g=nx.Graph()
                No=1
                # peak_max=0#z轴最高点
                # peak_min=1000
                # peak_max_no=1
                # peak_min_no=1
                for p in skel_points:
                    g.add_node(No,loc=p)#赋予位置属性loc
                    # if p[2]>peak_max:
                    #     peak_max=p[2]
                    #     peak_max_no=No#最高点序号
                    # if p[2]<peak_min:
                    #     peak_min=p[2]
                    #     peak_min_no=No#最高点序号
                    No+=1
                
                DisMap=[]#每个点对应的中心点
                surface={}#每个中心点对应的横截面
                print('3.calculate distance map...')

                for p in tqdm(all_points):#对所有非骨架点计算离其最近的骨架点，注意一定要是最大连通的，排除噪声点
                    if p not in skel_points:
                        DisMap.append({})
                        DisMap[-1]['loc']=p
                        min_dis=10000
                        min_no=None
                        for q in skel_points:
                            if in_neighbor(q,p,r=20):
                                dis=L2(p,q)
                                if dis<min_dis:
                                    min_dis=dis
                                    min_no=Loc2No(g,q)#这里已经排除中心线上的点              
                    DisMap[-1]['CenterPoint']=min_no 
                    DisMap[-1]['dis']=min_dis 
                    # print(p,min_no,min_dis)
                    if min_no not in surface.keys():
                        surface[min_no]=[]#找到的min_no是某一个骨架点
                    surface[min_no].append((p,min_dis))#以这个骨架点为中心构成一个截面

                print('4.complete graph...')    
                for k,v in tqdm(g._node.items()):
                    v['value']=mask[v['loc'][0],v['loc'][1],v['loc'][2]]#每个节点赋予CT值属性
                    v['neighbors']=[]#赋予属性neighbors，元素是邻居的序号
                    v['radius']=0#赋予半径属性
                    v['r_point']=None
                    if k in surface.keys() and len(surface[k])>0:
                        LCC_surface=LCC(k,surface,mask)#如果在直隶管辖区里面，避免游离的管辖点
                        for p,dis in LCC_surface:
                            if dis>v['radius']:
                                v['radius']=dis
                                v['r_point']=p
                    neighbors=get_neighbor(skeleton,v['loc'])
                    for loc in neighbors:
                        j=Loc2No(g,loc)
                        if  g.has_edge(k,j) or g.has_edge(j,k):
                            continue
                        else:
                            g.add_edge(k,j)
                            v['neighbors'].append(j)

                g = nx.minimum_spanning_tree(g,algorithm='prim')
                print('5.save data...')
                all_info['Graph']=g
                all_info['DisMap']=DisMap
                # all_info['peaks']=peaks
                with open(join(save_path,save_name),'wb') as f:
                    pickle.dump(all_info,f)
                # print(g)
                # break
            except:
                print('***************ERROR****************')
            


            
if __name__ == '__main__':
	main()