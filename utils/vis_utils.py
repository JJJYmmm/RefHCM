import cv2
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.link_pairs)):
            self.link_pairs[i].append(tuple(np.array(self.color[i])/255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i])/255.))
            
            
color2 = [(252,176,243),(252,176,243),(252,176,243),
    (0,176,240), (0,176,240), (0,176,240),
    (255,255,0), (255,255,0),(169, 209, 142),
    (169, 209, 142),(169, 209, 142),
    (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127)]

link_pairs2 = [
        [15, 13], [13, 11], [11, 5], 
        [12, 14], [14, 16], [12, 6], 
        [9, 7], [7,5], [5, 6], [6, 8], [8, 10],
        [3, 1],[1, 2],[1, 0],[0, 2],[2,4],
        ]


point_color2 = [(240,2,127),(240,2,127),(240,2,127), 
            (240,2,127), (240,2,127), 
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (252,176,243),(0,176,240),(252,176,243),
            (0,176,240),(252,176,243),(0,176,240),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142)]

chunhua_style = ColorStyle(color2, link_pairs2, point_color2)


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)
        
    return joints_dict


def vis_pose_result(data_numpy, pose_results, thickness, out_file):
    
    h = data_numpy.shape[0]
    w = data_numpy.shape[1]
        
    # Plot
    fig = plt.figure(figsize=(w/100, h/100), dpi=300)
    ax = plt.subplot(1,1,1)
    bk = plt.imshow(data_numpy[:,:,::-1])
    bk.set_zorder(-1)
    
    joints_dict = map_joint_dict(pose_results)
    
    # stick 
    for k, link_pair in enumerate(chunhua_style.link_pairs):

        if pose_results[link_pair[0], :2].sum() < 0.2 or pose_results[link_pair[1], :2].sum() < 0.2:
            continue
        
        if k in range(11,16):
            lw = thickness
        else:
            lw = thickness * 2

        line = mlines.Line2D(
                np.array([joints_dict[link_pair[0]][0],
                            joints_dict[link_pair[1]][0]]),
                np.array([joints_dict[link_pair[0]][1],
                            joints_dict[link_pair[1]][1]]),
                ls='-', lw=lw, alpha=1, color=link_pair[2],)
        line.set_zorder(0)
        ax.add_line(line)

    # black ring
    for k in range(pose_results.shape[0]):
        
        if pose_results[k, :2].sum() < 0.2:
            continue
        
        if k in range(5):
            radius = thickness
        else:
            radius = thickness * 2

        circle = mpatches.Circle(tuple(pose_results[k,:2]), 
                                    radius=radius, 
                                    ec='black', 
                                    fc=chunhua_style.ring_color[k], 
                                    alpha=1, 
                                    linewidth=1)
        circle.set_zorder(1)
        ax.add_patch(circle)
        
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)        
    plt.margins(0,0)

    buf = io.BytesIO() 
    plt.savefig(buf, format='png', bbox_inches='tight') 
    buf.seek(0)
    image = Image.open(buf)
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    plt.close()

    return image_cv