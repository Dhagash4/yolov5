import os
import click
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil


@click.command()
@click.option('--pickle_path',help="Path to pickle bbox file")
@click.option('--out_dir',help="Path to store labels(.txt)")
@click.option('--root_path',help="Root path to all images")


def gen_txt_labels(pickle_path,out_dir,root_path):

    data = pickle.load(open(pickle_path,'rb'))
    txt_file = os.path.dirname(os.path.abspath(root_path))
    train_file = open(os.path.join(txt_file,'train.txt'),"a+")
    val_file = open(os.path.join(txt_file,'val.txt'),"a+")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    idx = np.arange(len(data))
    np.random.shuffle(idx)
    train_idx = idx[:round(0.9*len(idx))]
    val_idx = idx[round(0.9*len(idx)):]
   
    data_np = list(data.keys())
    data_np = np.array(data_np)


    train = data_np[train_idx[:]]
    val = data_np[val_idx[:]]
    
    for k,v in data.items():
      
        
        image_path = os.path.join(root_path,k)
        h,w,_ = (cv2.imread(image_path)).shape
        labels = np.zeros((len(v),5))
        # img = cv2.imread(image_path)
        for j in  range(len(v)):

            class_id = 0
            x_center_norm = v[j][4] / w
            y_center_norm = v[j][5] / h
            #xmax - xmin / w
            width_norm = (v[j][2] - v[j][0]) / w
            #ymax - ymin / w
            height_norm = (v[j][3] - v[j][1]) / h

            labels[j] = np.array([class_id,x_center_norm,y_center_norm,width_norm,height_norm])

            point1 = (int(v[j][0]),int(v[j][3]))
            point2 = (int(v[j][2]),int(v[j][1]))

            # img = cv2.rectangle(img,point1,point2,(255,0,0),3)
        
        # cv2.imshow("window",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if k in train:

            train_label_path = "train/"+k.split('.')[0] + '.txt'
            train_file.write(f"{os.path.abspath(os.path.join(root_path,'train',k))}\n")
            if os.path.exists(os.path.join(out_dir,train_label_path)):
                print("label already exists")
            else:
                np.savetxt(os.path.join(out_dir,train_label_path),labels,['%d','%.3f','%.3f','%.3f','%.3f'])
            if os.path.exists(os.path.join(root_path,'train',k)):
                print("file already exists")
            else:
                shutil.copyfile(os.path.join(root_path,k),os.path.join(root_path,'train',k))
        else:

            val_label_path = "val/"+k.split('.')[0] + '.txt'
            val_file.write(f"{os.path.abspath(os.path.join(root_path,'val',k))}\n")
            if os.path.exists(os.path.join(out_dir,val_label_path)):
                print("label already exists")
            else:
                np.savetxt(os.path.join(out_dir,val_label_path),labels,['%d','%.3f','%.3f','%.3f','%.3f'])
            if os.path.exists(os.path.join(root_path,'val',k)):
                print("file already exists")
            else:
                shutil.copyfile(os.path.join(root_path,k),os.path.join(root_path,'val',k))
    
    train_file.close()
    val_file.close()



if __name__ == '__main__':
    gen_txt_labels()