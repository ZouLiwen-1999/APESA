'''
此脚本用于产生分割结果的Graph(包含半径等信息)和distance map的数据文件
保存在./pre_data/下面
'''
import nibabel as nib
from utils import *
from skimage.morphology import skeletonize
import networkx as nx
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import os
join=os.path.join
listdir=os.listdir
from math import sqrt
import pickle
import math
abs=math.fabs
from tqdm import tqdm
from sklearn.cluster import KMeans
#########超参数##############
suffix='nii'#可以指定哪个case来做，.nii.gz为全部
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

def vis_graph(g1,result,reverse):
    fig = plt.figure()
    plt.title('The Graph')
    ax = fig.add_subplot(projection='3d')
    if not reverse:
        R=1
    else:
        R=0
    for e in g1.edges():
        p1=g1.nodes[e[0]]['loc']
        r1=g1.nodes[e[0]]['radius']
        p2=g1.nodes[e[1]]['loc']
        r2=g1.nodes[e[1]]['radius']
        if result[int(r1)]==R and result[int(r2)]==R:
            ax.plot3D((p1[0],p2[0]),(p1[1],p2[1]),(p1[2],p2[2]),c='r') 
        else:
            ax.plot3D((p1[0],p2[0]),(p1[1],p2[1]),(p1[2],p2[2]),c='b') 
    plt.show()

def main():
    split_label=2
    all_info={}
    # all_info['Graph']=None
    # all_info['DisMap']=None
    # all_info['KeyPoints']={}#存放末端点，交叉点，最高点最低点等信息
    root='./predsTs_f0_v2_CA'#分割结果的mask路径
    save_path='./predsTs_f0_v2_CA_RC/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    showSkel=0
    NN=len(os.listdir(root))
    nn=0
    RC_file={}
    for file in listdir(root):
        nn+=1
        print('######',nn,'/',NN,file)
        # save_name=file.replace('.nii.gz','.pkl')#预处理数据保存名
        if 'IPMN_dingshiqing_A.nii' in file:  
            print('1. get skeleton...')
            nii=nib.load(join(root,file))
            mask=nii.get_fdata()
            LargestCC_mask = getLargestCC(mask).astype(int)#获取最大连通分量
            skeleton = skeletonize(mask)#利用Li的方法或者最大连通骨架
            skeleton=skeleton/np.max(skeleton)
            skel_points=mask2points(skeleton,label=1)
            all_points=mask2points(mask,label=1)#最大连通血管上的点
            # print(all_points)
            

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
                # if p not in skel_points:
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
                DisMap[-1]['CenterPoint']=min_no #p对应的骨架点序号
                DisMap[-1]['dis']=min_dis #p对应的距离
                # print(p,min_no,min_dis)
                if min_no not in surface.keys():
                    surface[min_no]=[]
                surface[min_no].append((p,min_dis))
            # print('surface')
            # print(surface.keys())
            # print(DisMap)

            print('4.complete graph...')  
            Rs=[]#存储胰管半径的列表
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
                Rs.append(v['radius'])
                neighbors=get_neighbor(skeleton,v['loc'])
                for loc in neighbors:
                    j=Loc2No(g,loc)
                    if  g.has_edge(k,j) or g.has_edge(j,k):
                        continue
                    else:
                        g.add_edge(k,j)
                        v['neighbors'].append(j)
            
            Rs=np.array(Rs).reshape(-1,1)
            # print(Rs)
            kmodel = KMeans(n_clusters = 2)
            kmodel.fit(Rs)
            kpredictions = kmodel.predict(Rs)

            #分别计算两类的平均半径
            avr0,avr1=[],[]
            result={}
            for r,p in zip(list(Rs),list(kpredictions)):
                if p==0:
                    avr0.append(r)
                else:
                    avr1.append(r)
                result[int(r)]=p
            reverse=False
            if sum(avr0)/len(avr0)>sum(avr1)/len(avr1):
                reverse=True

            max_r=max(sum(avr0)/len(avr0),sum(avr1)/len(avr1))
            min_r=min(sum(avr0)/len(avr0),sum(avr1)/len(avr1))
            RC=max_r/min_r
            RC_file[file]=RC
            print('Max/Min:',max_r/min_r)
            # save_name=file.replace('.nii','_'+str(max_r/min_r)+'.nii')
            g = nx.minimum_spanning_tree(g,algorithm='prim')
            #显示骨架
            if showSkel:
                print('We show you the skeleton points...')
                vis_graph(g,result,reverse)#显示中心线
            skel_result={}
            for k,v in tqdm(g._node.items()):
                r=v['radius']
                skel_result[k]=result[int(r)]

            #返回mask
            if RC>=2:#如果血管半径异常，则用聚类方法分开
                if reverse:
                    R=0
                else:
                    R=1
                for info in tqdm(DisMap):
                    loc=info['loc']
                    min_no=info['CenterPoint']
                    if min_no is not None:
                        output=skel_result[min_no]
                    else:
                        output=R-1
                    # print(output)
                    if output==R:
                        mask[loc[0],loc[1],loc[2]]=1
                    else:
                        mask[loc[0],loc[1],loc[2]]=split_label
            else:
                mask=np.uint8(mask*split_label)

            print('5.save data...')
            save_nii=nib.Nifti1Image(mask,nii.affine)
            nib.save(save_nii,join(save_path,file))


            # all_info['Graph']=g
            # all_info['DisMap']=DisMap
            # # all_info['peaks']=peaks
            with open('RC_file.pkl','wb') as f:
                pickle.dump(RC_file,f)
            # # print(g)
            # # break
            


            
if __name__ == '__main__':
	main()