import nibabel as nib
import scipy.io as scio
from skimage import measure
import numpy as np
import os
from skimage import morphology
import  matplotlib.pyplot as plt
import SimpleITK as sitk
from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import Line3D
from copy import deepcopy
from math import sqrt
# from matplotlib.pyplot import Line3D
# from node import Node
import json
exists=os.path.exists
join=os.path.join
from random import randint
import networkx as nx
from tqdm import tqdm

def NodeDist(s,d,graph):
    if s in graph[d]['neighbors']:
        return 1
    else:
        return float('inf')

def kruskal(graph):
    print('kruskal...')
    assert type(graph)==dict
    nodes = graph.keys()   
    visited = set()
    path = []
    next = None
    while len(visited) < len(nodes):
        distance = float('inf') 
        for s in nodes:
            for d in nodes:
                if s in visited and d in visited or s == d:
                    continue
                if  NodeDist(s,d,graph) < distance:
                    distance = NodeDist(s,d,graph)
                    pre = s
                    next = d
        path.append((pre, next))
        visited.add(pre)
        visited.add(next)
    print(path)
    return path

def show_path(path,graph):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for p in path:
        ax.plot3D((graph[p[0]]['loc'][0],graph[p[1]]['loc'][0]),(graph[p[0]]['loc'][1],graph[p[1]]['loc'][1]),(graph[p[0]]['loc'][2],graph[p[1]]['loc'][2]),c='b')
    plt.show()

def show_graph(graph):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    edge=0
    for k,v in graph.items():
        p1=v['loc']
        for q in v['neighbors']:
            p2=graph[q]['loc']
            ax.plot3D((p1[0],p2[0]),(p1[1],p2[1]),(p1[2],p2[2]),c='b')
            edge+=1  
    print('Node number:',len(graph))
    print('Edge number:',edge//2)
    plt.show()

#图的BFS广度优先遍历
def bfsTravel(graph,source=1):
    print('traveling the graph...')
    # 传入的参数为邻接表存储的图和一个开始遍历的源节点
    frontiers = [source]     # 表示前驱节点
    travel = [source]       # 表示遍历过的节点
    # 当前驱节点为空时停止遍历
    while frontiers:        
        nexts = []          # 当前层的节点（相比frontier是下一层）
        for frontier in frontiers:
            for current in graph[frontier]['neighbors']: # 遍历当前层的节点
                if current not in travel:   # 判断是否访问过
                    travel.append(current)  # 没有访问过则入队
                    nexts.append(current)   # 当前结点作为前驱节点
        frontiers = nexts   # 更改前驱节点列表
    print(travel)
    return travel

def find_keypoints(graph):
    print('find keypoints...')
    keypoints={'endpoints':[],'junctions':[],'all':[]}
    for k,v in graph.items():
        if len(v['neighbors'])==1:
            keypoints['endpoints'].append(k)
            keypoints['all'].append(k)
        elif len(v['neighbors'])>2:
            keypoints['junctions'].append(k)
            keypoints['all'].append(k)
    print(keypoints)
    return keypoints


def getLargestCC(segmentation):#获得最大连通分量
    mask = measure.label(segmentation)
    largestCC = mask == np.argmax(np.bincount(mask.flat)[1:])+1
    return largestCC

def getLCC_zlw(segmentation):#获得最大连通分量
    masks,lcc_num=measure.label(segmentation,background=0,return_num=True,connectivity=3)
    if lcc_num == 0 or lcc_num == 1:
        return segmentation
    else:
        max_num=0
        for i in range(1, lcc_num+1):  #注意这里的范围，为了与连通域的数值相对应
            # 计算面积，保留最大面积对应的索引标签，然后返回二值化最大连通域
            if np.sum(masks == i) > max_num:
                max_num = np.sum(masks == i)
                max_label = i
        return (masks == max_label).astype(np.uint8)

def mask2points(mask,label=1):
    '''
    将骨架表示成节点列表
    '''
    positions = np.argwhere(mask==label)
    # deepth, height, weight = mask.shape
    #
    points = []
    No=0
    for position in positions:
        currentPoint =(position[0],position[1],position[2])
        points.append(currentPoint)
    return points

def get_neighbor(mask,loc,label=1):
    #得到节点p的neighbors
    deepth, height, weight = mask.shape
    I,J,K=loc
    neighbors=[]
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                if min(I+i,J+j,K+k)<0 or I+i>=deepth or J+j>=height or K+k>=weight or (i==j==k==0):
                    continue
                elif mask[I+i,J+j,K+k]==label:
                    # neighbor_num+=1
                    neighbors.append((I+i,J+j,K+k))
    return neighbors

def loc2id(G,loc):
    for k,v in G._node.items():
        if v['loc']==loc:
            return k

def vis_graph(g):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for e in g.edges():
        p1=g.nodes[e[0]]['loc']
        # for q in v['neighbors']:
        p2=g.nodes[e[1]]['loc']
        ax.plot3D((p1[0],p2[0]),(p1[1],p2[1]),(p1[2],p2[2]),c='r')
        # edge+=1  
    # print('Node number:',len(graph))
    # print('Edge number:',edge//2)
    plt.show()

def vis_branchI(g,branchI_names,add_point=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    color_bar=['g','b','y','c','m','w','k','coral','cyan','purple']

    for e in g.edges():#显示所有边
        p1=g.nodes[e[0]]['loc']
        p2=g.nodes[e[1]]['loc']
        ax.plot3D((p1[0],p2[0]),(p1[1],p2[1]),(p1[2],p2[2]),c='b')
    n=0
    for name, branch in branchI_names.items():
        xs,ys,zs = [],[],[]
        for _ in branch:
            p=g.nodes[_]['loc']
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
            color=color_bar[n%len(color_bar)]
        n+=1
        ax.scatter(xs,ys,zs,s=5,c=color,marker='o')
    if add_point is not None:
        ax.scatter(add_point[0],add_point[1],add_point[2],s=8,c='r',marker='*')
    plt.show()

def vis_branchII(g,branchII):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for e in g.edges():#显示所有边
        p1=g.nodes[e[0]]['loc']
        # for q in v['neighbors']:
        p2=g.nodes[e[1]]['loc']
        ax.plot3D((p1[0],p2[0]),(p1[1],p2[1]),(p1[2],p2[2]),c='b')
    
    color_bar={'Ao':'r','CA':'b','LGA':'g','SA':'m','CHA':'c','SMA':'k'}
    for name,vessel in branchII.items():
        xs,ys,zs = [],[],[]
        for _ in vessel:
            p=g.nodes[_]['loc']
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
        ax.scatter(xs,ys,zs,s=5,c=color_bar[name],marker='o') 
    plt.show()

def vis_branchI_mask(tree,points_info,mask,all_points):
    #显示一级血管分支
    deepth, height, weight = mask.shape
    for i in range(deepth):
        for j in range(height):
            for k in range(weight):
                if mask[i,j,k]==1:
                    min_point=min_dis(tree,i,j,k)
                    # min_point=min_dis2(i,j,k,all_points,mask)
                    if min_point in points_info.keys():
                        mask[i,j,k]=points_info[min_point]
                        print((i,j,k),'->',mask[i,j,k])
                    # else:
                    #     print((i,j,k),'->',mask[i,j,k])
    return mask

def min_dis(tree,i,j,k):
    #求中心线tree上和mask上（i,j,k）最近点的坐标
    Min=10000
    p=None
    for s in list(tree._node.keys()):
        dis=L2((i,j,k),tree.nodes[s]['loc'])
        if dis<Min:
            p=s
            Min=dis
        if dis<3:
            p=s
            break
    return p

def min_dis2(I,J,K,all_points,mask):
    r=2
    deepth, height, weight = mask.shape
    for i in range(-1*r,r+1):
        for j in range(-1*r,r+1):
            for k in range(-1*r,r+1):
                if min(I+i,J+j,K+k)<0 or I+i>=deepth or J+j>=height or K+k>=weight or (i==j==k==0):
                    continue
                elif (I+i,J+j,K+k) in all_points:
                    return (I+i,J+j,K+k)

def min_dis_v3(tree,i,j,k,points_info):
    #求中心线tree上和mask上（i,j,k）最近点的坐标
    #对tree上的AO降低一下门槛
    Min=10000
    p=None
    for s in list(tree._node.keys()):
        dis=L2((i,j,k),tree.nodes[s]['loc'])
        #开个挂嘻嘻嘻
        if points_info[s]==1:
            dis=dis-5
        if dis<Min:
            p=s
            Min=dis
        if dis<3:
            p=s
            break
    return p

    

def L2(p1,p2):
    assert len(p1)==len(p2)
    d2=0
    for i in range(len(p1)):
        d2+=(p1[i]-p2[i])**2
    return sqrt(d2)
    
def vis_branchs(branchs,tree):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    A=0
    for branch in branchs:
        color=branch_color(A)
        A+=1
        for i in range(len(branch)-1):
            p1=tree.nodes[branch[i]]['loc']
            p2=tree.nodes[branch[i+1]]['loc']
            ax.plot3D((p1[0],p2[0]),(p1[1],p2[1]),(p1[2],p2[2]),c=color)
    plt.show()

def branch_color(A):
    tp=A%7
    if tp==0:
        color='b'
    elif tp==1:
        color='y'
    elif tp==2:
        color='g'
    elif tp==3:
        color='m'
    elif tp==4:
        color='k'
    elif tp==5:
        color='c'
    else:
        color='b'
    return color

def show_skeleton(points):
    # #中心线可视化
    # points=self.points
    fig = plt.figure()
    # print('Start to show the skleton of'+self.file+' ...')
    ax = fig.add_subplot(projection='3d')
    xs,ys,zs = [],[],[]
    # skel_points=self.mask2point(self.skeleton,label=1)
    # print(np.unique(self.skeleton))
    print('Number of points on the skeleton:',len(points))
    for p in points:
        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])
    ax.scatter(xs,ys,zs,s=5,c='b',marker='o')
    plt.show()

def plan1(AO,branchI,AO_nodes,endpoints,junctions,tree,nii,mask,points,save_path,file,visBranch,returnMask):
    '''
    正常解剖顺序:
    找到腹腔干，胃左动脉，肝总动脉和脾动脉
    ''' 
    print('#################plan1##################') 
    CAs=branchI[0]#腹腔干血管总支
    k0=AO_nodes[0]#腹腔干起点
    print('k0:',k0)
    print('degree of k0:',tree.degree(k0))
    ed1=set(endpoints)&CAs#腹腔干血管总支上的末端节点
    #找到最左边和左右边的点
    x_max=0#z轴最高点
    x_min=1000
    x_max_no=0
    x_min_no=0
    for p in ed1:
        if tree.nodes[p]['loc'][0]>x_max:
            x_max=tree.nodes[p]['loc'][0]
            x_max_no=p   
        if tree.nodes[p]['loc'][0]<x_min:
            x_min=tree.nodes[p]['loc'][0]
            x_min_no=p
    sp=x_max_no#脾动脉分支最右端
    lp=x_min_no#肝部动脉做左端

    path_right=set(nx.dijkstra_path(tree,k0,sp))#从腹腔干到脾动脉的路径
    path_left=set(nx.dijkstra_path(tree,k0,lp))#从腹腔干到肝部动脉的路径
    path_common=path_right&path_left&set(junctions)#重合部分的交叉点
    path_common=sorted(path_common, key=lambda keynode : tree.nodes[keynode]['loc'][0],reverse=True) 
    print(path_common)
    assert len(path_common)>=2#理论上是三个交叉点
    if len(path_common)>2:
        k1,k2=path_common[1],path_common[2]#胃左动脉交叉点
        CA=set(nx.dijkstra_path(tree,k0,k1))#腹腔干
        LGA=set()#胃左动脉
        for p in ed1:
            path=set(nx.dijkstra_path(tree,p,k1))#往k1跑
            if len(path&{k1,k2})==1:
                LGA=LGA|path
            
        SA=set(nx.dijkstra_path(tree,k1,sp))#脾动脉
        CHA=set(nx.dijkstra_path(tree,k2,lp))#肝总动脉
        branchII={'Ao':AO,'CA':CA,'LGA':LGA,'SA':SA,'CHA':CHA,'SMA':branchI[1]}
        if visBranch:
            vis_branchII(tree,branchII)
    else:#如果只有两个交叉点，说明没有LGA
        k1=list(set(path_common)-{k0})[0]#胃左动脉交叉点
        CA=set(nx.dijkstra_path(tree,k0,k1))#腹腔干
        LGA=set()#胃左动脉
        SA=set(nx.dijkstra_path(tree,k1,sp))#脾动脉
        CHA=set(nx.dijkstra_path(tree,k1,lp))#肝总动脉
        branchII={'Ao':AO,'CA':CA,'LGA':LGA,'SA':SA,'CHA':CHA,'SMA':branchI[1]}
        if visBranch:
            vis_branchII(tree,branchII)
    branchII_labels={'Ao':2,'CA':3,'LGA':4,'SA':5,'CHA':6,'SMA':7}

    if returnMask:
        #将中心线上的分段结果可视化到原mask上
        # all_points=set(tree._node.keys())
        # other_branches=all_points-AO-branchI
        points_info={}
        N=len(branchI)
        for k in list(tree._node.keys()):
            for name,vessel in branchII.items():
                if k in vessel:
                    points_info[k]=branchII_labels[name]
                    break        
        print('We show you the points label:')
        print(points_info)     
        labeled_mask=vis_branchI_mask(tree,points_info,mask,points)
        save_nii = nib.Nifti1Image(labeled_mask, nii.affine)
        nib.save(save_nii, join(save_path,file.replace('.nii.gz','_vsl2.nii.gz')))


def plan2(AO,branchI,AO_nodes,endpoints,junctions,tree,nii,mask,points,save_path,file,visBranch,returnMask):
    #CA和SMA长到一起
    #找到腹腔干，胃左动脉，肝总动脉和脾动脉
    print('#################plan2##################')
    CA_SMAs=branchI[0]#CA_SMA总支
    # k0=AO_nodes[0]#腹腔干起点
    assert len(list(CA_SMAs&AO))==1
    k0=list(CA_SMAs&AO)[0]
    print('k0:',k0)
    print('degree of k0:',tree.degree(k0))
    ed0=set(endpoints)&CA_SMAs#CA_SMA总支上的末端节点

    #找到最下边的点
    z_min=1000
    z_min_no=0
    for p in ed0:  
        if tree.nodes[p]['loc'][2]<z_min:
            z_min=tree.nodes[p]['loc'][2]
            z_min_no=p
    # print('z_min_no:',z_min_no)
    bottom_path=nx.dijkstra_path(tree,z_min_no,k0)#注意顺序！！！
    # print('bottom_path:',bottom_path)
    # print('junctions:',junctions)
    bottom_path_junc=[]
    for i in bottom_path:
        if i in junctions:
            bottom_path_junc.append(i)

    k1=bottom_path_junc[-2]#
    print('k1:',k1)
    print('degree of k1:',tree.degree(k1))
    CA=nx.dijkstra_path(tree,k0,k1)
    print('CA:',CA)

    SMA=set()
    CAs=set()
    for k in ed0:#对CA_SMA上每个末端点
        path=set(nx.dijkstra_path(tree,k,k1))#求每个末端点到k1的最短路
        print(path&set(bottom_path))
        if len(path&set(bottom_path))==0:#
            CAs=CAs|path#
        else:
            SMA=SMA|path
    print('SMA:',SMA)
    print('CAs:',CAs)
    ed1=set(endpoints)&CAs#腹腔干血管总支上的末端节点
    #找到最左边和左右边的点
    x_max=0#z轴最高点
    x_min=1000
    x_max_no=0
    x_min_no=0
    for p in ed1:
        if tree.nodes[p]['loc'][0]>x_max:
            x_max=tree.nodes[p]['loc'][0]
            x_max_no=p   
        if tree.nodes[p]['loc'][0]<x_min:
            x_min=tree.nodes[p]['loc'][0]
            x_min_no=p
    sp=x_max_no#脾动脉分支最右端
    lp=x_min_no#肝部动脉做左端
    sp=x_max_no#脾动脉分支最右端
    lp=x_min_no#肝部动脉做左端

    path_right=set(nx.dijkstra_path(tree,k1,sp))#从腹腔干到脾动脉的路径
    path_left=set(nx.dijkstra_path(tree,k1,lp))#从腹腔干到肝部动脉的路径
    path_common=path_right&path_left&set(junctions)#重合部分的交叉点
    path_common=sorted(path_common, key=lambda keynode : tree.nodes[keynode]['loc'][0],reverse=True) 
    assert len(path_common)>=2#理论上是三个交叉点
    if len(path_common)>2:
        k2,k3=path_common[1],path_common[2]#胃左动脉交叉点
        LGA=set()#胃左动脉
        for p in ed1:
            path=set(nx.dijkstra_path(tree,p,k2))#往k2跑
            if len(path&{k2,k3})==1:
                LGA=LGA|path
            
        SA=set(nx.dijkstra_path(tree,k2,sp))#脾动脉
        CHA=set(nx.dijkstra_path(tree,k3,lp))#肝总动脉
        branchII={'Ao':AO,'CA':CA,'LGA':LGA,'SA':SA,'CHA':CHA,'SMA':SMA}
        if visBranch:
            vis_branchII(tree,branchII)
    else:
        k2=list(set(path_common)-{k1})[0]#胃左动脉交叉点
        LGA=set()#胃左动脉
        SA=set(nx.dijkstra_path(tree,k2,sp))#脾动脉
        CHA=set(nx.dijkstra_path(tree,k2,lp))#肝总动脉
        branchII={'Ao':AO,'CA':CA,'LGA':LGA,'SA':SA,'CHA':CHA,'SMA':SMA}
        if visBranch:
            vis_branchII(tree,branchII)
    branchII_labels={'Ao':2,'CA':3,'LGA':4,'SA':5,'CHA':6,'SMA':7}

    if returnMask:
        #将中心线上的分段结果可视化到原mask上
        # all_points=set(tree._node.keys())
        # other_branches=all_points-AO-branchI
        points_info={}
        N=len(branchI)
        for k in list(tree._node.keys()):
            for name,vessel in branchII.items():
                if k in vessel:
                    points_info[k]=branchII_labels[name]
                    break        
        print('We show you the points label:')
        print(points_info)     
        labeled_mask=vis_branchI_mask(tree,points_info,mask,points)
        save_nii = nib.Nifti1Image(labeled_mask, nii.affine)
        nib.save(save_nii, join(save_path,file.replace('.nii.gz','_vsl2.nii.gz')))

def return_mask_v1_2(tree,points_info,mask):
    width,height,deepth = mask.shape
    # all_n=deepth*height*width
    # no=0
    for i in tqdm(range(width)):
        for j in range(height):
            for k in range(deepth):
                if mask[i,j,k]==1:
                    # min_point=min_dis(tree,i,j,k)
                    min_point=min_dis_v3(tree,i,j,k,points_info)
                    if min_point in points_info.keys():
                        mask[i,j,k]=points_info[min_point]
                        # no+=1
                        # print((i,j,k),'->',mask[i,j,k])
                # if no
    return mask

def get_branch4point(tree,p,AO_nodes,AO):
    for k in AO_nodes:
        path=set(nx.dijkstra_path(tree,p,k))
        if len(path&AO)==1:
            return k   

def LCC(k,surface,mask):
    mask_=np.zeros_like(mask,np.uint8)
    for point,_ in surface[k]:
        mask_[point[0],point[1],point[2]]=1
    LCC=getLCC_zlw(mask_)
    new_surface=[]
    for p,v in surface[k]:
        if LCC[p[0],p[1],p[2]]==1:
            new_surface.append((p,v))
    return new_surface