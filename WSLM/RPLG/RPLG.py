import nibabel as nib
# from zmq import Again
from utils import *
# from skimage import morphology
import networkx as nx
import warnings
import os
warnings.filterwarnings("ignore")
join=os.path.join
ld=os.listdir
import pickle
#########超参数##############
returnMask=1
visBranch=0
suffix='nii'#
unsuffix='_0000'
again=1
############################

def main():
    root=''
    pre_path=''
    save_path=''
    LargestCC_save_path=''
    for path in [save_path,LargestCC_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    NN=len(ld(root))
    nn=0
    for file in ld(root):
        nn+=1
        print('##',nn,'/',NN,file)
        if suffix in file and file not in ld(save_path) and file.replace('.nii.gz','.pkl') in ld(pre_path):
            try:
                #加载预处理数据
                with open(join(pre_path,file.replace('.nii.gz','.pkl')),'rb') as f:
                    data=pickle.load(f)        
                
                nii=nib.load(join(root,file))
                mask=nii.get_fdata()
                mask=(mask>0).astype(np.uint8)
                mask = getLargestCC(mask).astype(int)#获取最大连通分量,否则无法构建连通图
                # LargestCC_mask = getLargestCC(mask).astype(int)#获取最大连通分量
                LargestCC_nii = nib.Nifti1Image(mask, nii.affine)
                nib.save(LargestCC_nii, join(LargestCC_save_path,file))
                #获取Graph和最高的和最低的点
                tree=data['Graph']#加载预处理的最小生成树
        
                #提取关键节点
                endpoints=[]
                junctions=[]
                for d in tree.degree():
                    if d[1]==1:
                        endpoints.append(d[0])
                    elif d[1]>2:
                        junctions.append(d[0])

                #sort函数默认从小到大，把末端点按照z轴上下高度排列
                up_down=sorted(endpoints, key=lambda k : tree.nodes[k]['loc'][2],reverse=False) 
                print('data shape:',mask.shape)
                print('top z:',up_down[-1],tree.nodes[up_down[-1]]['loc'][2],'||bottom z:',up_down[0],tree.nodes[up_down[0]]['loc'][2])


                #在最低的10个末端点中选最粗的2个
                peak_min_no=0
                max_r=0
                for k in up_down[:10]:#最低的若干个点中取最粗的一个
                    print(k,tree.nodes[k]['loc'][2],tree.nodes[k]['radius'])
                    if tree.nodes[k]['radius']>max_r:
                        peak_min_no=k
                        max_r=tree.nodes[k]['radius']
                print('bottom_no:',peak_min_no,'bottom_no_r:',max_r)
                

                peak_min_no2=0#找第二个
                max_r2=0
                for k in up_down[:10]:#最低的若干个点中取最粗的一个
                    if k!=peak_min_no:
                        if tree.nodes[k]['radius']>max_r2:
                            peak_min_no2=k
                            max_r2=tree.nodes[k]['radius']
                print('bottom_no2:',peak_min_no2,'bottom_no_r:',max_r2)

                peak_min_no3=0#找第三个
                max_r3=0
                for k in up_down[:10]:#最低的若干个点中取最粗的一个
                    if k!=peak_min_no and k!=peak_min_no2:
                        if tree.nodes[k]['radius']>max_r3:
                            peak_min_no3=k
                            max_r3=tree.nodes[k]['radius']
                print('bottom_no3:',peak_min_no3,'bottom_no_r:',max_r3)
                # return

                no_IAs=False
                if max_r2<0.5*max_r:#如果找到的第二粗的底端半径不到第一粗的一半则判断为没有下髂动脉
                    no_IAs=True

                #在最高的10个末端点中选最粗的1个
                peak_max_no=0
                max_r=0
                for k in up_down[-10:]:
                    if tree.nodes[k]['radius']>max_r:
                        peak_max_no=k
                        max_r=tree.nodes[k]['radius']
                print('top_no:',peak_max_no,'top_no_r:',max_r)

                #找腹主动脉AO
                if not no_IAs:
                    print('Maybe there is IAs...')
                    AO=set(nx.dijkstra_path(tree,peak_max_no, peak_min_no))&set(nx.dijkstra_path(tree,peak_max_no, peak_min_no2))
                    # AO=set(nx.dijkstra_path(tree,peak_max_no, peak_min_no))&set(nx.dijkstra_path(tree,peak_max_no, peak_min_no2))&set(nx.dijkstra_path(tree,peak_max_no, peak_min_no3))
                else:
                    print('Maybe there is no IAs...')
                    AO=set(nx.dijkstra_path(tree,peak_max_no, peak_min_no))
                    print('##',peak_min_no)
                # return
                print('AO length:',len(AO))

                #找到AO上的分叉点
                AO_nodes=list(set(junctions)&AO)

                # 通过z轴坐标从高到底将交叉点排序
                AO_nodes=sorted(AO_nodes, key=lambda keynode : tree.nodes[keynode]['loc'][2],reverse=True) 
                
                print('Find',len(AO_nodes),'junctions on the AO...')

                branchI={}#一级血管分支
                order=1
                for k in AO_nodes:#对AO上每个分叉点k，找到他的分支集合
                    if tree.degree(k)==3:#如果只有一个分支出来                    
                        branch=set()
                        for p in endpoints:
                            path=set(nx.dijkstra_path(tree,p,k))
                            if len(path&AO)==1:
                                branch=branch|path  
                    
                        if len(branch)<10:#去掉很短的分支
                            AO=AO|branch
                            print('remove No.',k,'branch!') 
                        else:
                            branchI[k]={}
                            branchI[k]['vessel']=branch
                            if order>=4:
                                branchI[k]['order']=4#其他分支用label4表示 
                            else:
                                order+=1
                                branchI[k]['order']=order

                    else:#有两个或以上的分支出来
                        nbrs=tree[k]#得到k的所有邻居点

                        for nbr in nbrs:
                            if nbr in AO_nodes:
                                nbrs.remove(nbr)
                                
                        print('****There is/are',len(nbrs),'sub-branches from',k,'point on the AO!****')
                        # assert len(nbrs)==2#只处理两个分支的情况
                        
                        #以y方向的变量作为排序标准
                        nbrs=sorted(nbrs, key=lambda keynode : abs(tree.nodes[k]['loc'][1]-tree.nodes[keynode]['loc'][1]),reverse=True)

                        for nbr in nbrs:
                            if nbr not in AO_nodes:#如果邻居点不在AO上 
                                branch=set()
                                for p in endpoints:
                                    path=set(nx.dijkstra_path(tree,p,nbr))
                                    if len(path & (AO|{nbr})  )==1:
                                        branch=branch|path  
                                branch=branch|{k}
                                if len(branch)<10:#去掉很短的分支
                                    AO=AO|branch
                                    print('remove No.',k,'branch!') 
                                else:
                                    branchI[nbr]={}
                                    branchI[nbr]['vessel']=branch
                                    branchI[nbr]['order']=order 
                                    if order>=4:
                                        branchI[k]['order']=4#其他分支用label4表示 
                                    else:
                                        order+=1
                                        branchI[k]['order']=order


                print('Now there are',len(branchI),'branches on the AO...')

                branchI_names={'Ao':AO}
                branchI_labels={'Ao':1,'CA':2,'SMA':3,'Others':4,'SA':5,'CHA':6,'LGA':7,'GDA':8,'PHA':9}
                branchI_names['Others']=set()
                for k,v in branchI.items():
                    if v['order']==2:
                        CA_root=k
                        CAs=v['vessel']
                    elif v['order']==3:
                        branchI_names['SMA']=v['vessel']
                    elif v['order']==4:
                        branchI_names['Others']=branchI_names['Others']|v['vessel']

                CA_endpoints=[]
                for p in endpoints:
                    if p in CAs:
                        CA_endpoints.append(p)

                left_right=sorted(CA_endpoints, key=lambda k : tree.nodes[k]['loc'][0],reverse=False) 
                left_point,right_point=left_right[0],left_right[-1]


                left_path=set(nx.dijkstra_path(tree,left_point,CA_root))
                right_path=set(nx.dijkstra_path(tree,right_point,CA_root))
                
                CA=left_path&right_path   
                CHAs=left_path-CA
                SA=right_path-CA
                LGA=set()
                for p in CA_endpoints:
                    path=set(nx.dijkstra_path(tree,p,CA_root))
                    if len(path&CHAs)>2:
                        CHAs=CHAs|(path-CA)
                    elif len(path&SA)>2:
                        SA=SA|(path-CA)
                    else:
                        LGA=LGA|(path-CA)

                CHA_endpoints=[]
                for p in endpoints:
                    if p in CHAs:
                        CHA_endpoints.append(p)

                
                CHA_points=sorted(CHA_endpoints, key=lambda k : tree.nodes[k]['loc'][2],reverse=False) 
                GDA_point=CHA_points[0]
                GDA_path=set(nx.dijkstra_path(tree,GDA_point,CA_root))

                CHA=(GDA_path&left_path)-CA
                # print('GDA_path----------->',GDA_path)
                print('CA----------->',CA)
                GDA=GDA_path-left_path
                # print('GDA----------->',GDA)
                # GDA=GDA-CHA
                # print('GDA----------->',GDA)
                for p in CHA_endpoints:
                    path=set(nx.dijkstra_path(tree,p,CA_root))
                    if len(path&GDA)>2:
                        GDA=GDA|(path-left_path)
                PHA=CHAs-CHA-GDA

                print('GDA----------->',GDA)
                branchI_names['CA']=CA
                branchI_names['CHA']=CHA
                branchI_names['SA']=SA
                branchI_names['LGA']=LGA
                branchI_names['GDA']=GDA
                branchI_names['PHA']=PHA


                if visBranch:        
                    add_point=tree.nodes[peak_min_no2]['r_point']
                    vis_branchI(tree,branchI_names,add_point)
                
                if returnMask:
                    points_info={}
                    for p in list(tree._node.keys()):
                        in_branch=False
                        for name,branch in branchI_names.items():
                            if p in branch:
                                points_info[p]=branchI_labels[name]
                                in_branch=True
                                break  
                        if not in_branch:#如果有些中心线点不在任何branch上，可能是上面的末端最短路漏了
                            print('point',p ,'not in any branch before, we will find where they are...')
                            k=get_branch4point(tree,p,AO_nodes,AO)
                            assert k is not None
                            points_info[p]=branchI[k]['vessel']
                    # print(len(list(tree._node.keys())),len(points_info))      
                    print('We show you the points label:')
                    print(set(points_info.values()))

                    labeled_mask=return_mask_v1_2(tree,points_info,mask)
                    save_nii = nib.Nifti1Image(labeled_mask, nii.affine)
                    nib.save(save_nii, join(save_path,file))
            except:
                print('********ERROR*****************')    

if __name__=='__main__':
    main()

        