import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
plt.rcParams["figure.figsize"] = 10, 8
plt.rcParams["font.size"     ] = 14

import numpy as np

import sys
sys.path.append('../src') 

from gi import get_gamma_position_in_pixels

from torchvision import transforms

# General plots

def histoplot(var, xlabel, ylabel, bins=100, figsize=(4,4), title=""):
    """
    Histogram variable var and return contents and bins
    """
    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    h = plt.hist(var,bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return h[0],h[1]


def scatter_xyz(x,y,z, figsize=(4, 4)):
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    axs[0].scatter(x, y, alpha=0.7, edgecolor='k')
    axs[0].set_title('x vs y')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    axs[1].scatter(x, z, alpha=0.7, edgecolor='k')
    axs[1].set_title('x vs z')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('z')

    axs[2].scatter(y, z, alpha=0.7, edgecolor='k')
    axs[2].set_title('y vs z')
    axs[2].set_xlabel('y')
    axs[2].set_ylabel('z')

    plt.tight_layout()
    plt.show()

# Plots GI (gamma interaction df)

def scatter_xyze(df, figsize=(4, 4)):
    fig, axis = plt.subplots(2, 2, figsize=figsize)
    axs = axis.ravel()
    
    axs[0].scatter(df.x1, df.x2, alpha=0.7, edgecolor='k')
    axs[0].set_title('x1 vs x2')
    axs[0].set_xlabel('x1')
    axs[0].set_ylabel('x2')

    axs[1].scatter(df.y1, df.y2, alpha=0.7, edgecolor='k')
    axs[1].set_title('y1 vs y2')
    axs[1].set_xlabel('y1')
    axs[1].set_ylabel('y2')

    axs[2].scatter(df.z1, df.z2, alpha=0.7, edgecolor='k')
    axs[2].set_title('z1 vs z2')
    axs[2].set_xlabel('z1')
    axs[2].set_ylabel('z2')

    axs[3].scatter(df.e1, df.e2, alpha=0.7, edgecolor='k')
    axs[3].set_title('e1 vs e2')
    axs[3].set_xlabel('e1')
    axs[3].set_ylabel('e2')
    
    plt.tight_layout()
    plt.show()


def plot_amplitude(df, num_bins = 20, xmin=2e+4, xmax=4e+4, figsize=(6, 4), title=""):
    energies = df.groupby("event").sum().amplitude.values
    
    fig, ax0 = plt.subplots(1, 1, figsize=figsize)
    h = ax0.hist(energies, num_bins, (xmin, xmax))
    ax0.set_xlabel("Event energy")
    ax0.set_ylabel('Events/bin')
    ax0.set_title(title)
    
    fig.tight_layout()
    return h[0],h[1]


def plot_image(dfq, dfg,  evtsel, x_spatial, y_spatial, figsize=(6, 6)):
    
    image = select_image_from_df(dfq,evtsel)
    plot_image2(image, dfg,  evtsel, x_spatial, y_spatial, figsize=figsize)
    

def plot_image2(image, dfg,  evtsel, x_spatial, y_spatial, figsize=(6, 6)):
    
    charge_matrix = image
    xt1, yt1,xt2, yt2 =get_gamma_position_in_pixels(dfevt, x_spatial, y_spatial, debug)
    xt1, yt1,xt2, yt2 =get_gamma_position(dfg, evtsel, x_spatial, y_spatial)
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot the image with imshow (with pixel axis from 0 to 8)
    img = ax1.imshow(charge_matrix.T, extent=[0, 8, 0, 8], origin='lower', aspect='auto',
                    cmap='viridis', interpolation='none')

    # Add colorbar for the imshow plot
    cbar = fig.colorbar(img, ax=ax1, pad=0.18)
    cbar.set_label('Charge')

    # Create a secondary x-axis (spatial) that matches the pixel axis
    ax2 = ax1.twiny()
    ax2.set_xlim(x_spatial[0], x_spatial[-1])
    ax2.set_xlabel('X')

    # Create a secondary y-axis (spatial) that matches the pixel axis
    ax3 = ax1.twinx()
    ax3.set_ylim(y_spatial[0], y_spatial[-1])
    ax3.set_ylabel('Y')

    # Show the plot

    ax1.scatter(xt1, yt1,  facecolor='red')
    ax1.scatter(xt2, yt2,  facecolor='blue')

    ax1.text(2, 6, f"Event = {evtsel}", color='white', fontsize=12, ha='center', va='center')

    plt.tight_layout()
    plt.show()


## Plots with images

def plot_image(img, meta, x_spatial, y_spatial, figsize=(6, 4), debug=False):
    """
    Plots  one image
    
    """
        
    fig, axs = plt.subplots(1, 1,figsize=figsize)            
    xt1, yt1,xt2, yt2 =get_gamma_position_in_pixels(meta, x_spatial, y_spatial, debug)

    # Plot the image with imshow (with pixel axis from 0 to 8)
    axs.imshow(img.T, extent=[0, 8, 0, 8], origin='lower', aspect='auto',
                        cmap='viridis', interpolation='none')

    axs.scatter(xt1, yt1,  facecolor='red')
    axs.scatter(xt2, yt2,  facecolor='blue')
    axs.text(2, 6, f"Event = {int(meta.event)}", color='white', fontsize=12, ha='center', va='center')

    plt.tight_layout()
    plt.show()


def plot_images(img, dfg, dfs, img_numbers, figsize=(14, 4), debug=False):
    """
    Plots 8 images with the gamma points
    
    """
    
    if len(img_numbers) > 8:
        print("range too large, can plot a max of 8 images")
        return 0

    x_spatial = dfs.sensor_x.values
    y_spatial = dfs.sensor_y.values
        
    fig, axs = plt.subplots(2, 4,figsize=figsize)        
    ftx = axs.ravel()
    
    for i, ev in enumerate(img_numbers): 
        charge_matrix = img[ev]
        dfevt = dfg.iloc[ev]
        evnt = dfevt.event    

        if debug:
            print(f"plotting image number {ev}, event number = {evnt}")
    
        xt1, yt1,xt2, yt2 =get_gamma_position_in_pixels(dfevt, x_spatial, y_spatial, debug)

        if debug:
            print(charge_matrix)


        # Plot the image with imshow (with pixel axis from 0 to 8)
        ftx[i].imshow(charge_matrix.T, extent=[0, 8, 0, 8], origin='lower', aspect='auto',
                           cmap='viridis', interpolation='none')

        ftx[i].scatter(xt1, yt1,  facecolor='red')
        ftx[i].scatter(xt2, yt2,  facecolor='blue')
        ftx[i].text(2, 6, f"Event = {evnt}", color='white', fontsize=12, ha='center', va='center')

        
    plt.tight_layout()
    plt.show()


# Plots cnn 

def plot_image_ds(dataset, indx):
    img = dataset[indx][0]
    plt.rcParams["figure.figsize"] = 4, 4
    plt.imshow(transforms.ToPILImage()(img))


def plot_images_ds(dataset, imgs=(0,8), sx=2, figsize=(14, 4)):
    
    lenx = imgs[1] - imgs[0]
    sx = sx
    sy = int(np.ceil(lenx/sx))
        
    fig, axs = plt.subplots(sx, sy,figsize=figsize)        
    ftx = axs.ravel()
    for  i in range(*imgs):        
        img = dataset[i][0]
        ftx[i].imshow(transforms.ToPILImage()(img))


def plotxyz_twoc(tdl, nbins=50):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax = axes.ravel()
    
    ax[0].hist(tdl.delta_x_NN, bins=nbins, 
             label=f"x ($\sigma$ = {np.std(tdl.delta_x_NN):.2f})", alpha=0.7)
    ax[0].set_xlabel("NN (xtrue - xpredicted)",fontsize=14)
    ax[0].set_ylabel("Counts/bin",fontsize=14)
    ax[0].legend()
    
    ax[1].hist(tdl.delta_y_NN, bins=nbins, 
             label=f"y ($\sigma$ = {np.std(tdl.delta_y_NN):.2f})", alpha=0.7)
    ax[1].set_xlabel("NN (ytrue - ypredicted)",fontsize=14)
    ax[1].set_ylabel("Counts/bin",fontsize=14)
    ax[1].legend()
    
    ax[2].hist(tdl.delta_z_NN, bins=nbins, 
             label=f"z ($\sigma$ = {np.std(tdl.delta_z_NN):.2f})", alpha=0.7)
    ax[2].set_xlabel("NN (ztrue - zpredicted)",fontsize=14)
    ax[2].set_ylabel("Counts/bin",fontsize=14)
    ax[2].legend()

    ax[3].hist(tdl.delta_x_NN2, bins=nbins, 
             label=f"x ($\sigma$ = {np.std(tdl.delta_x_NN2):.2f})", alpha=0.7)
    ax[3].set_xlabel("NN (xtrue2 - xpredicted2)",fontsize=14)
    ax[3].set_ylabel("Counts/bin",fontsize=14)
    ax[3].legend()
    
    ax[4].hist(tdl.delta_y_NN2, bins=nbins, 
             label=f"y ($\sigma$ = {np.std(tdl.delta_y_NN2):.2f})", alpha=0.7)
    ax[4].set_xlabel("NN (ytrue2 - ypredicted2)",fontsize=14)
    ax[4].set_ylabel("Counts/bin",fontsize=14)
    ax[4].legend()
    
    ax[5].hist(tdl.delta_z_NN2, bins=nbins, 
             label=f"z ($\sigma$ = {np.std(tdl.delta_z_NN2):.2f})", alpha=0.7)
    ax[5].set_xlabel("NN (ztrue2 - zpredicted2)",fontsize=14)
    ax[5].set_ylabel("Counts/bin",fontsize=14)
    ax[5].legend()
    
    fig.tight_layout()
    plt.show()


def plotxyz(tdl, nbins=50):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    flat_axes = axes.ravel()
    ax0, ax1, ax2 = flat_axes[0], flat_axes[1], flat_axes[2]
    
    ax0.hist(tdl.delta_x_NN, bins=nbins, 
            label=f"x ($\sigma$ = {np.std(tdl.delta_x_NN):.2f})", alpha=0.7)
    ax0.set_xlabel("NN (xtrue - xpredicted)",fontsize=14)
    ax0.set_ylabel("Counts/bin",fontsize=14)
    ax0.legend()
    ax1.hist(tdl.delta_y_NN, bins=nbins, 
            label=f"y ($\sigma$ = {np.std(tdl.delta_y_NN):.2f})", alpha=0.7)
    ax1.set_xlabel("NN (ytrue - ypredicted)",fontsize=14)
    ax1.set_ylabel("Counts/bin",fontsize=14)
    ax1.legend()
    ax2.hist(tdl.delta_z_NN, bins=nbins, 
            label=f"z ($\sigma$ = {np.std(tdl.delta_z_NN):.2f})", alpha=0.7)
    ax2.set_xlabel("NN (ztrue - zpredicted)",fontsize=14)
    ax2.set_ylabel("Counts/bin",fontsize=14)
    ax2.legend()
    fig.tight_layout()
    plt.show()


def plot_loss(epochs, train_losses, val_losses,figsize=(10, 4)):
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    xvals_train = np.arange(0,epochs,1)
    xvals_val = np.arange(0,epochs,1)
    axes.plot(xvals_train,train_losses,label='training')
    axes.plot(xvals_val,val_losses,label='validation')
    axes.set_ylabel("Loss")
    axes.set_xlabel("epochs")
    axes.legend()
    fig.tight_layout()
    plt.show()


