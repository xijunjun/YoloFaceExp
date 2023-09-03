from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import os
import numpy as np
import shutil
import platform
import math

############路径及窗口大小设置############################
# local_path = u"Z:\workspace\yoloface\dataset\crawler"          #图像根目录
local_path=r'Z:\workspace\yoloface\dataset\label_test'

MAX_WIDTH=1800                    #图像窗口最大宽度
MAX_HEIGHT=1000                    #图像窗口最大高度
########################################

global_facerctpts_label=[]
global_facerctpts_label_offset=[]
global_facelandpts_label=[]
global_flags=[0,0,0,0,0] 
#cursor_rect,cursor_land,curstate,
# 方框的光标，关键点的光标，当前窗口处于哪个阶段。
global_face_label_list=[]
#

local_path=local_path+'/'
key_dic={}
def load_key_val():
    key_val_path='key_val.txt'
    if 'Windows' in platform.system():
        key_val_path='key_val_win.txt'
    lines=open(key_val_path).readlines()
    for line in lines:
        item=line.split(' ')
        vals=item[1].split(',')
        val_lst=[]
        for val in vals:
            val_lst.append(int(val))
        key_dic[item[0]]=val_lst
        # print item[0],val_lst
load_key_val()

def limit_imgw(img):
    if  img.shape[1]>360:
        img=cv2.resize( img,(360,int(360.0/img.shape[1]*img.shape[0])),interpolation=cv2.INTER_CUBIC )
    if  img.shape[0]>360:
        img=cv2.resize( img,(int(360.0/img.shape[0]*img.shape[1]),360),interpolation=cv2.INTER_CUBIC )
    return  img


def limit_window(disimg,winname):
    wm_ratio=1.0
    if disimg.shape[1] > MAX_WIDTH or disimg.shape[0] > MAX_HEIGHT:
        if (disimg.shape[1] / float(disimg.shape[0])) > (MAX_WIDTH / float(MAX_HEIGHT)):
            cv2.resizeWindow(winname, MAX_WIDTH, int(MAX_WIDTH / float(disimg.shape[1]) * disimg.shape[0]))
            wm_ratio = MAX_WIDTH / float(disimg.shape[1])
        else:
            cv2.resizeWindow(winname, int(MAX_HEIGHT / float(disimg.shape[0]) * disimg.shape[1]), MAX_HEIGHT)
            wm_ratio = MAX_HEIGHT / float(disimg.shape[0])
    else:
        cv2.resizeWindow(winname, disimg.shape[1], disimg.shape[0])
    return wm_ratio

def resize_facecrop_window(disimg,winname):
    wm_ratio=1.0
    TARGET_WIDTH=800
    TARGET_HEIGHT=800
    # if disimg.shape[1] > TARGET_WIDTH or disimg.shape[0] > TARGET_HEIGHT:
    if (disimg.shape[1] / float(disimg.shape[0])) > (TARGET_WIDTH / float(TARGET_HEIGHT)):
        cv2.resizeWindow(winname, TARGET_WIDTH, int(TARGET_WIDTH / float(disimg.shape[1]) * disimg.shape[0]))
        wm_ratio = TARGET_WIDTH / float(disimg.shape[1])
    else:
        cv2.resizeWindow(winname, int(TARGET_HEIGHT / float(disimg.shape[0]) * disimg.shape[1]), TARGET_HEIGHT)
        wm_ratio = TARGET_HEIGHT / float(disimg.shape[0])
    # else:
    #     cv2.resizeWindow(winname, disimg.shape[1], disimg.shape[0])
    return wm_ratio


def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

def find_nearest_point_index(x, y, pts_label_all):
    min_distance = float('inf')  # 初始化一个无限大的距离
    nearest_index = None

    for i, point in enumerate(pts_label_all):
        # 计算欧几里得距离
        distance = math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
        
        # 如果当前点的距离小于最小距离，则更新最小距离和最近点的索引
        if distance < min_distance:
            min_distance = distance
            nearest_index = i
    
    return nearest_index



###########################
def get_facecrop_rect(img,pt1,pt2):
    h,w,c=img.shape
    pts2=np.array([np.array(pt1),np.array(pt2)])
    pt_ct=pts2.mean(axis=0)
    pts_res=pts2-pt_ct
    pts_res[:,0]*=1.5 #x
    pts_res[:,1]*=1.5 #y
    pts_res+=pt_ct
    # print(pts2)
    # print(pt_ct)
    pts_res=pts_res.astype(np.int32)
    pts_res[:,0]=np.clip(pts_res[:,0],0,w-1)
    pts_res[:,1]=np.clip(pts_res[:,1],0,h-1)
    return pts_res


def find_nearest_rct_index(x, y, face_label_list):
    min_distance = float('inf')  # 初始化一个无限大的距离
    nearest_index = 0

    for i,face_label in enumerate(face_label_list):
        rect_label,faceland_label=face_label
        distance =0
        for point in rect_label:
            distance += math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_index = i

    return nearest_index



def update_view_main():
    disimg=np.array(img)
    for pt in global_facerctpts_label:
        cv2.circle(disimg, (pt[0], pt[1]),int(2), (0, 0, 255), thickness=-1)

    if len(global_facerctpts_label)==2:
        # rect_cursor_ind=global_flags[0]
        # pt_rect_cur=global_facerctpts_label[rect_cursor_ind]
        # cv2.circle(disimg, (pt_rect_cur[0], pt_rect_cur[1]),int(3), (0, 0, 255), thickness=-1)

        tl=global_facerctpts_label[0]
        br=global_facerctpts_label[1]   
        cv2.rectangle(disimg,(tl[0],tl[1]),(br[0],br[1]), (0, 0, 255), 1)
        
        bigrct=get_facecrop_rect(disimg,tl,br)
        tl=bigrct[0]
        br=bigrct[1] 
        # cv2.rectangle(disimg,(tl[0],tl[1]),(br[0],br[1]), (0, 255, 255), 1)

    if len(global_face_label_list)>0:
        for face_label in global_face_label_list:
            rect_label,faceland_label=face_label
            for pt in faceland_label:
                cv2.circle(disimg, (pt[0], pt[1]),int(3), (255, 0, 255), thickness=-1)

            tl=rect_label[0]
            br=rect_label[1]   
            cv2.rectangle(disimg,(tl[0],tl[1]),(br[0],br[1]), (0, 0, 255), 1)

        rect_cursor=global_flags[0]
        rect_label,faceland_label=global_face_label_list[rect_cursor]
        tl,br=rect_label[0],rect_label[1]  
        cv2.rectangle(disimg,(tl[0],tl[1]),(br[0],br[1]), (0, 0, 255), 3)


    cv2.imshow('img', disimg)
    limit_window(disimg,'img')


def update_view_facecrop():
    global global_facecroped
    # tl=global_facerctpts_label[0]
    # br=global_facerctpts_label[1]  
    # croprct=get_facecrop_rect(img,tl,br)
    # tlx,tly,brx,bry=croprct[0][0],croprct[0][1],croprct[1][0],croprct[1][1]
    # facecroped=img[tly:bry+1,tlx:brx+1,:].copy()

    disimg=np.array(global_facecroped)

    tl=global_facerctpts_label[0]
    br=global_facerctpts_label[1]   
    cv2.rectangle(disimg,(tl[0],tl[1]),(br[0],br[1]), (0, 0, 255), 1)
    for pt in global_facerctpts_label:
        cv2.circle(disimg, (pt[0], pt[1]),int(2), (0, 0, 255), thickness=-1)


    for pt in global_facelandpts_label:
        cv2.circle(disimg, (pt[0], pt[1]),int(2), (0, 0, 255), thickness=-1)

    pt_cursor=global_flags[1]
    if len(global_facelandpts_label)==5:
        if pt_cursor>=2:
            pt=global_facelandpts_label[pt_cursor-2]
            cv2.circle(disimg, (pt[0], pt[1]),int(4), (0, 0, 255), thickness=-1)
        else:
            pt=global_facerctpts_label[pt_cursor]
            cv2.circle(disimg, (pt[0], pt[1]),int(4), (0, 0, 255), thickness=-1)


    cv2.imshow('refine_face',disimg)
    resize_facecrop_window(disimg,'refine_face')


def mouse_func_facerect(event,x,y,flags,param):
    # if len(global_facerctpts_label) == 2:
    #     return
    # global_flags
    if global_flags[2]!=0:
        return
    if event == cv2.EVENT_LBUTTONDOWN:

        if len(global_facerctpts_label) < 2:
            # print(x,y)
            global_facerctpts_label.append([x,y])
            # update_view_main()
        if len(global_facerctpts_label)==2:
            # get4pts()
            # get_info()
            update_view_main()
            refine_face()

        update_view_main()
        cv2.waitKey(1)#注意此处等待按键

    if event == cv2.EVENT_MOUSEMOVE:
        global_flags[0]=find_nearest_rct_index(x, y, global_face_label_list)
        update_view_main()
        cv2.waitKey(1)#注意此处等待按键


def mouse_func_faceland(event,x,y,flags,param):
    global global_facecroped

    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(global_facelandpts_label) < 5:
            # print(x,y)
            global_facelandpts_label.append([x,y])
            # update_view_main()
        elif len(global_facelandpts_label)==5:
            # get4pts()
            # get_info()
            # refine_face()
            pt_cursor=global_flags[1]
            if pt_cursor>=2:
                global_facelandpts_label[pt_cursor-2]=[x,y]
            else:
                global_facerctpts_label[pt_cursor]=[x,y]
            pass

        update_view_facecrop()
        cv2.waitKey(1)#注意此处等待按键

    if event == cv2.EVENT_MOUSEMOVE:
        pts_label_all=list(global_facerctpts_label)
        pts_label_all.extend(global_facelandpts_label)
        # print(len(pts_label_all))
        # pts_label_all=np.array(pts_label_all)
        global_flags[1]=find_nearest_point_index(x, y, pts_label_all)
        
        update_view_facecrop()
        cv2.waitKey(1)#注意此处等待按键


# 

# global_facelandpts_label
def refine_face():
    global global_facecroped
    global global_facerctpts_label

    global_flags[2]=1

    # cropimg=np.array(img)
    tl=global_facerctpts_label[0]
    br=global_facerctpts_label[1]  
    croprct=get_facecrop_rect(img,tl,br)
    tlx,tly,brx,bry=croprct[0][0],croprct[0][1],croprct[1][0],croprct[1][1]
    global_facecroped=img[tly:bry+1,tlx:brx+1,:].copy()


    facerctpts_label=np.array(global_facerctpts_label)
    facerctpts_label[:,0]-=tlx
    facerctpts_label[:,1]-=tly
    global_facerctpts_label=list(facerctpts_label)


    cv2.namedWindow('refine_face', cv2.WINDOW_FREERATIO)
    # cv2.imshow('refine_face',global_facecroped)
    update_view_facecrop()
    cv2.setMouseCallback('refine_face', mouse_func_faceland)

    # update_view_main()
    breakflag=0
    while 1:
        key=cv2.waitKey(0)
        if key in key_dic['SPACE']:
            global_flags[0]=1-global_flags[0]
        if key in key_dic['ENTER']:
            facelandpts_label_tofull=np.array(global_facelandpts_label)
            facelandpts_label_tofull[:,0]+=tlx
            facelandpts_label_tofull[:,1]+=tly

            facerctpts_label_tofull=np.array(global_facerctpts_label)
            facerctpts_label_tofull[:,0]+=tlx
            facerctpts_label_tofull[:,1]+=tly
            # global_facerctpts_label=list(facerctpts_label_tofull)

            global_face_label_list.append((list(facerctpts_label_tofull),list(facelandpts_label_tofull)))

            cv2.destroyWindow('refine_face')
            global_facelandpts_label.clear()
            global_facerctpts_label.clear()
            global_flags[2]=0
            breakflag=1
            break
        if key in key_dic['BACK']:
            if len(global_facelandpts_label)>0:
                global_facelandpts_label.pop()

        if breakflag!=1:
            # update_view_main()
            update_view_facecrop()





if __name__ == '__main__':
    ims=get_ims(local_path)

    cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
    cv2.setMouseCallback('img', mouse_func_facerect)

    for i,im in enumerate(ims):
        # if i==0:
        #     continue
        print(im)

        img=cv2.imread(im)
        cv2.imshow('img',img)
        limit_window(img,'img')

        global_facelandpts_label.clear()
        global_facerctpts_label.clear()

        while 1:
            key=cv2.waitKey()
            print(key)
            if key==27:
                exit(0)
            if key in key_dic['ENTER']:
                break
            if key in key_dic['BACK']:
                if len(global_facerctpts_label)>0:
                    global_facerctpts_label.pop()
                    # print('pop')
            if key in key_dic['DELETE']:
                if len(global_face_label_list)>0:
                    rect_cursor=global_flags[0]
                    del global_face_label_list[rect_cursor]
                    global_flags[0]=0
                print('delete')


            update_view_main()

        global_face_label_list.clear()