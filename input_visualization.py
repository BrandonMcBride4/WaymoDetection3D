
import torch
import numpy as np
from matplotlib import pyplot as plt

world_space_offset = torch.tensor([-75.2,-75.2,-2])
output_voxel_size = 75.2*2/193

input_voxel_size = 75.2*2/1088

voxel_spacing = torch.Tensor([.1,.1,.15])


#ground truth expressed as [n,8] where 8 is [x,y,z,xw,yw,zw,dir,class]
def visualize_ground_truth(ground_truth, point_cloud,foreground,voxels,gt_reg,gt_class, gt_dir,colors):



  ax = plt.subplot(221,projection='3d')


  ax2 = plt.subplot(222)

  ax3 = plt.subplot(223)

  ax4 = plt.subplot(224,projection='3d')

  active_output = torch.nonzero(torch.any(gt_reg.to(torch.bool),dim=0))
  

  c = torch.zeros((point_cloud.shape[0],3))

  # c[~foreground,2] = 1
  c[foreground,0] = 1
  c[foreground,1] = 0
  c[foreground,2] = 0

  ax2.scatter(point_cloud[:,0].numpy(), point_cloud[:,1].numpy(), point_cloud[:,2].numpy(),c=c,linewidths=.2)

  voxels = voxels/100
  ax3.scatter(voxels[:,2].numpy(), voxels[:,1].numpy(), voxels[:,0].numpy(),linewidths=.2)



  for gt in ground_truth:
    plot3d_box(gt,ax,colors)
    plot2d_box(gt,ax2,colors)

  create_BEV_vis(gt_reg, gt_dir, gt_class,ax4,colors)

  ax.set_xlim(-60,60)
  ax.set_ylim(-60,60)
  ax.set_zlim(0,120)

  ax4.set_xlim(-60,60)
  ax4.set_ylim(-60,60)
  ax4.set_zlim(0,120)

  ax.figure.set_size_inches(10,10)

  ax2.set_xlim(-60,60)
  ax2.set_ylim(-60,60)
  # ax2.set_zlim(-40,40)

  ax.set_xlabel('x axis')
  ax.set_ylabel('y axis')

  ax2.set_xlabel('x axis')
  ax2.set_ylabel('y axis')

  ax3.set_xlabel('x axis')
  ax3.set_ylabel('y axis')

  ax4.set_xlabel('x axis')
  ax4.set_ylabel('y axis')

  ax.set_title('Ground Truth Boxes')
  ax2.set_title('Point Cloud with Highlighted Interior Points')
  ax3.set_title('Input Voxels')
  ax4.set_title('BEV Output')

def plot3d_box(box,ax,colors):
    x,y = torch.meshgrid((torch.tensor([-box[3],box[3]]),torch.tensor([-box[4],box[4]])))

    rot_x = torch.cos(box[6])*x - torch.sin(box[6])*y
    rot_y = torch.sin(box[6])*x + torch.cos(box[6])*y

    plot_x = rot_x + box[0]
    plot_y = rot_y + box[1]

    z = box[2] + torch.ones(plot_x.shape)*box[5] 
    z_neg = box[2] - torch.ones(plot_x.shape)*box[5] 


    ax.plot_wireframe(plot_x.numpy(),plot_y.numpy(),z.numpy(),color=colors[box[7].to(torch.int).item()],linewidths=.2)
    ax.plot_wireframe(plot_x.numpy(),plot_y.numpy(),z_neg.numpy(),color=colors[box[7].to(torch.int).item()],linewidths=.2)

    ax.plot(np.array([plot_x[0,0],plot_x[0,0]]),np.array([plot_y[0,0],plot_y[0,0]]) , np.array([box[2]+box[5],box[2]-box[5]]),color=colors[box[7].to(torch.int).item()],linewidth=.2)
    ax.plot(np.array([plot_x[1,0],plot_x[1,0]]),np.array([plot_y[1,0],plot_y[1,0]]) , np.array([box[2]+box[5],box[2]-box[5]]),color=colors[box[7].to(torch.int).item()],linewidth=.2)
    ax.plot(np.array([plot_x[0,1],plot_x[0,1]]),np.array([plot_y[0,1],plot_y[0,1]]) , np.array([box[2]+box[5],box[2]-box[5]]),color=colors[box[7].to(torch.int).item()],linewidth=.2)
    ax.plot(np.array([plot_x[1,1],plot_x[1,1]]),np.array([plot_y[1,1],plot_y[1,1]]) , np.array([box[2]+box[5],box[2]-box[5]]),color=colors[box[7].to(torch.int).item()],linewidth=.2)

def plot2d_box(box,ax,colors):
    x,y = torch.meshgrid((torch.tensor([-box[3],box[3]]),torch.tensor([-box[4],box[4]])))

    rot_x = torch.cos(box[6])*x - torch.sin(box[6])*y
    rot_y = torch.sin(box[6])*x + torch.cos(box[6])*y

    plot_x = rot_x + box[0]
    plot_y = rot_y + box[1]

    z = box[2] + torch.ones(plot_x.shape)*box[5] 
    z_neg = box[2] - torch.ones(plot_x.shape)*box[5] 

    ax.plot(np.array([plot_x[0,0],plot_x[1,0],plot_x[1,1],plot_x[0,1], plot_x[0,0]]) , np.array([plot_y[0,0],plot_y[1,0],plot_y[1,1],plot_y[0,1], plot_y[0,0]]),color=colors[box[7].to(torch.int).item()],linewidth=.5)


def create_BEV_vis(gt_reg, gt_dir, gt_class,ax,colors):
  for x in range(193):
    for y in range(193):
      max_class = torch.argmax(gt_class[:,x,y].to(torch.int))
      if(max_class != 0):
        box = gt_reg[:,x,y]     
 
        box = torch.cat([box, torch.tensor([max_class-1])])
        box[:3] = box[:3] + (torch.tensor([x,y,0])+.5)* output_voxel_size + world_space_offset
        temp = box[0].clone()
        box[0] = box[1]
        box[1] = temp
        dir = -1 if torch.argmax(gt_dir[:,x,y].to(torch.int)) == 0 else 1

        box[6] = box[6]*dir
        plot3d_box(box,ax,colors)
  


#outward facing methods

def visualize_batch_input(batch,colors={0:'r',1:'b',2:'g'}):
  visualize_ground_truth(batch['gt_label'],batch['point_cloud'],batch['foreground'],batch['voxels'],batch['gt_reg'],batch['gt_class'],batch['gt_dir'],colors)
  plt.show()
def visualize_network_output(gt_reg, gt_dir, gt_class,colors={0:'r',1:'b',2:'g'}):
  ax = plt.subplot(111,projection='3d')
  create_BEV_vis(gt_reg, gt_dir, gt_class,ax,colors)
  ax.set_xlim(-60,60)
  ax.set_ylim(-60,60)
  ax.set_zlim(0,120)

  ax.set_xlabel('x axis')
  ax.set_ylabel('y axis')

  ax.set_title('BEV Output')
  ax.figure.set_size_inches(10,10)


  plt.show()


