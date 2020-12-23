
import torch
import numpy as np
from matplotlib import pyplot as plt

world_space_offset = torch.tensor([-75.2,-75.2,-2])
output_voxel_size = torch.tensor([75.2*2/193,75.2*2/193,6])

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

  print(voxels)
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
    box = box.cpu()
    x,y = torch.meshgrid((torch.tensor([-box[3]/2,box[3]/2]),torch.tensor([-box[4]/2,box[4]/2])))

    rot_x = torch.cos(box[6])*x - torch.sin(box[6])*y
    rot_y = torch.sin(box[6])*x + torch.cos(box[6])*y

    plot_x = rot_x + box[0]
    plot_y = rot_y + box[1]

    z = box[2] + torch.ones(plot_x.shape)*box[5]/2 
    z_neg = box[2] - torch.ones(plot_x.shape)*box[5]/2 


    ax.plot_wireframe(plot_x.numpy(),plot_y.numpy(),z.numpy(),color=colors[box[7].to(torch.int).item()],linewidths=.2)
    ax.plot_wireframe(plot_x.numpy(),plot_y.numpy(),z_neg.numpy(),color=colors[box[7].to(torch.int).item()],linewidths=.2)

    ax.plot(np.array([plot_x[0,0],plot_x[0,0]]),np.array([plot_y[0,0],plot_y[0,0]]) , np.array([box[2]+box[5]/2,box[2]-box[5]/2]),color=colors[box[7].to(torch.int).item()],linewidth=.2)
    ax.plot(np.array([plot_x[1,0],plot_x[1,0]]),np.array([plot_y[1,0],plot_y[1,0]]) , np.array([box[2]+box[5]/2,box[2]-box[5]/2]),color=colors[box[7].to(torch.int).item()],linewidth=.2)
    ax.plot(np.array([plot_x[0,1],plot_x[0,1]]),np.array([plot_y[0,1],plot_y[0,1]]) , np.array([box[2]+box[5]/2,box[2]-box[5]/2]),color=colors[box[7].to(torch.int).item()],linewidth=.2)
    ax.plot(np.array([plot_x[1,1],plot_x[1,1]]),np.array([plot_y[1,1],plot_y[1,1]]) , np.array([box[2]+box[5]/2,box[2]-box[5]/2]),color=colors[box[7].to(torch.int).item()],linewidth=.2)

def plot2d_box(box,ax,colors):
    x,y = torch.meshgrid((torch.tensor([-box[3]/2,box[3]/2]),torch.tensor([-box[4]/2,box[4]/2])))

    rot_x = torch.cos(box[6])*x - torch.sin(box[6])*y
    rot_y = torch.sin(box[6])*x + torch.cos(box[6])*y

    plot_x = rot_x + box[0]
    plot_y = rot_y + box[1]

    z = box[2] + torch.ones(plot_x.shape)*box[5]/2 
    z_neg = box[2] - torch.ones(plot_x.shape)*box[5]/2 

    ax.plot(np.array([plot_x[0,0],plot_x[1,0],plot_x[1,1],plot_x[0,1], plot_x[0,0]]) , np.array([plot_y[0,0],plot_y[1,0],plot_y[1,1],plot_y[0,1], plot_y[0,0]]),color=colors[box[7].to(torch.int).item()],linewidth=.5)


def create_BEV_vis(gt_reg, gt_dir, gt_class,ax,colors, thresh = .5):
  dev = gt_reg.device
  max_value, max_class = torch.max(gt_class, dim = 0 )
  x,y = torch.meshgrid(torch.arange(193),torch.arange(193))
  mask = (max_class > 0) & (max_value > thresh)

  x = x[mask].to(dev)
  y = y[mask].to(dev)


  box = gt_reg[:,mask]
  box = torch.vstack([box,(max_class[mask]-1).unsqueeze(0)])
  
  box[:3] = box[:3] + (torch.vstack([x.unsqueeze(0),y.unsqueeze(0),torch.zeros(x.shape).unsqueeze(0).to(dev)])+.5)*output_voxel_size.unsqueeze(1).to(dev) + world_space_offset.unsqueeze(1).to(dev)
  box[[0,1],:] = box[[1,0],:]
  box[6] = box[6] * (torch.argmax(gt_dir[:,mask].to(torch.int),dim=0)-0.5)*2

  for b in box.T:
    plot3d_box(b,ax,colors)


  


#outward facing methods

def visualize_batch_input(batch, colors={0:'r',1:'b',2:'g'}, num = 0):
  visualize_ground_truth(batch['gt_label'][num],batch['point_cloud'][num],batch['foreground'][num],batch['voxel_coords'][batch['voxel_coords'][:,0]==num][:,1:],batch['gt_reg'][num],batch['gt_class'][num],batch['gt_dir'][num],colors)
  plt.show()
def visualize_network_output(batch, pred_reg, pred_dir, pred_class,colors={0:'r',1:'b',2:'g'},num = 0, thresh = .5):
  
  ax = plt.subplot(111,projection='3d')
  create_BEV_vis(pred_reg[num].permute((2,0,1)), pred_dir[num].permute((2,0,1)), pred_class[num].permute((2,0,1)),ax,colors,thresh)
  create_BEV_vis(batch['gt_reg'][num], batch['gt_dir'][num], batch['gt_class'][num], ax, {0:'k',1:'k',2:'k'}, thresh)
  ax.set_xlim(-60,60)
  ax.set_ylim(-60,60)
  ax.set_zlim(0,120)

  ax.set_xlabel('x axis')
  ax.set_ylabel('y axis')

  ax.set_title('BEV Output')
  ax.figure.set_size_inches(10,10)

  plt.show()

def plot_bbox_output(boxes, colors={0:'r',1:'b',2:'g'}, num = 0):
  ax = plt.subplot(111,projection='3d')

  for b in boxes[num]:
    plot3d_box(b,ax,colors)
  ax.set_xlim(-60,60)
  ax.set_ylim(-60,60)
  ax.set_zlim(0,120)

  ax.set_xlabel('x axis')
  ax.set_ylabel('y axis')

  ax.set_title('BEV Output')
  ax.figure.set_size_inches(10,10)

  plt.show()





