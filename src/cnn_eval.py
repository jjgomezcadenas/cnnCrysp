import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from collections import namedtuple


def single_run(train_loader, device, model, optimizer, criterion):
    """
    Run the model for a single event

    """
    print(f"** Run for 1 event**")

    for epoch in range(1):
        print(f"epoch = {epoch}")
    
        for i, (images, labels) in enumerate(train_loader):  
            if i>1: break
            print(f"i = {i}")
            print(f"images = {images.shape}")
            print(f"labels = {labels.shape}")
            images = images.to(device)
            labels = labels.to(device)
            
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)

            print(f"outputs = {outputs.data.shape}")
           
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            
            loss.backward()
            optimizer.step()
    
            print(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")


def train_cnn(train_loader, val_loader, model, optimizer, device, criterion, 
              batch_size, epochs=10, iprnt=100):
    """
    train the CNN
    """

    print(f"Training with  ->{len(train_loader)*batch_size} images")
    print(f"size of train loader  ->{len(train_loader)} images")
    print(f"Evaluating with  ->{len(val_loader)*batch_size} images")
    print(f"size of eval loader  ->{len(val_loader)} images")
    print(f"Running for epochs ->{epochs}")

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_losses_epoch, val_losses_epoch = [], []

        #logging.debug(f"\nEPOCH {epoch}")
        print(f"\nEPOCH {epoch}")

        # Training step
        for i, (images, positions) in enumerate(train_loader):

            images = images.to(device)
            positions = positions.to(device)

            model.train()  #Sets the module in training mode.
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = model(images) # image --> model --> (x,y,z)
            loss = criterion(outputs, positions) # compare labels with predictions
            loss.backward()  # backward pass
            optimizer.step()
            
            train_losses_epoch.append(loss.data.item())
            
            if((i+1) % (iprnt * batch_size) == 0):
                print(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")

        #print(f"Done with training after epoch ->{epoch}")
        #print(f"Start validations in epoch ->{epoch}")
        
        # Validation step
        with torch.no_grad():  #gradients do not change
            model.eval()       # Sets the module in evaluation mode.
            
            for i, (images, positions) in enumerate(val_loader):

                images = images.to(device)
                positions = positions.to(device)

                outputs = model(images)
                loss = criterion(outputs, positions)

                val_losses_epoch.append(loss.data.item())
                if((i+1) % (iprnt * batch_size) == 0):
                    print(f"Validation Step {i + 1}/{len(val_loader)}, Loss: {loss.data.item()}")

        #print(f"Done with validation after epoch ->{epoch}")
        train_losses.append(np.mean(train_losses_epoch))
        val_losses.append(np.mean(val_losses_epoch))
        print(f"--- EPOCH {epoch} AVG TRAIN LOSS: {np.mean(train_losses_epoch)}")
        print(f"--- EPOCH {epoch} AVG VAL LOSS: {np.mean(val_losses_epoch)}")
    
    #logging.info(f"Out of loop after epoch ->{epoch}")
    return train_losses, val_losses


def evaluate_cnn(test_loader, model, device, twoc=0):
    """
    evaluate the CNN returning the difference between true and predicted for the three coordinates

    """
    true_x, true_y, true_z = [],[],[]
    if twoc > 0:
        true_x2, true_y2, true_z2 = [],[],[]
    if twoc == 2:
        true_e = []
        true_e2 = []

    predicted_x, predicted_y, predicted_z = [],[],[]
    if twoc > 0:
        predicted_x2,  predicted_y2,  predicted_z2 = [],[],[]
    if twoc == 2:
        predicted_e = []
        predicted_e2 = []
         
    with torch.no_grad():

        model.eval()
        for i, (images, positions) in enumerate(test_loader):

            images = images.to(device)
            outputs = model(images).cpu()

            for x in positions[:,0]: true_x.append(x)
            for y in positions[:,1]: true_y.append(y)
            for z in positions[:,2]: true_z.append(z)
            if twoc > 0:
                for x in positions[:,3]: true_x2.append(x)
                for y in positions[:,4]: true_y2.append(y)
                for z in positions[:,5]: true_z2.append(z)
            
            if twoc == 2:
                for e in positions[:,6]: true_e.append(e)
                for e in positions[:,7]: true_e2.append(e)

            for x in outputs[:,0]: predicted_x.append(x)
            for y in outputs[:,1]: predicted_y.append(y)
            for z in outputs[:,2]: predicted_z.append(z)
                
            if twoc > 0:
                for x in outputs[:,3]: predicted_x2.append(x)
                for y in outputs[:,4]: predicted_y2.append(y)
                for z in outputs[:,5]: predicted_z2.append(z)
            if twoc == 2:
                for e in outputs[:,6]: predicted_e.append(e)
                for e in outputs[:,7]: predicted_e2.append(e)

    # Convert to numpy arrays
    true_x = np.array(true_x); true_y = np.array(true_y); true_z = np.array(true_z)
    if twoc >0:
        true_x2 = np.array(true_x2); true_y2 = np.array(true_y2); true_z2 = np.array(true_z2)

    if twoc == 2:
        true_e = np.array(true_e)
        true_e2 = np.array(true_e2)

    predicted_x = np.array(predicted_x) 
    predicted_y = np.array(predicted_y); predicted_z = np.array(predicted_z)
    if twoc > 0:
       predicted_x2 = np.array(predicted_x2) 
       predicted_y2 = np.array(predicted_y2); predicted_z2 = np.array(predicted_z2)
    if twoc == 2:
       predicted_e = np.array(predicted_e)
       predicted_e2 = np.array(predicted_e2)

    # Compute deltas for the NN.
    delta_x_NN = true_x - predicted_x
    delta_y_NN = true_y - predicted_y
    delta_z_NN = true_z - predicted_z

    if twoc>0:
        delta_x_NN2 = true_x2 - predicted_x2
        delta_y_NN2 = true_y2 - predicted_y2
        delta_z_NN2 = true_z2 - predicted_z2
    if twoc == 2:
        delta_e_NN = true_e - predicted_e
        delta_e_NN2 = true_e2 - predicted_e2

    if twoc == 2:
        tdeltas = namedtuple('tdeltas',
            'delta_x_NN, delta_y_NN, delta_z_NN, delta_e_NN, delta_x_NN2, delta_y_NN2, delta_z_NN2, delta_e_NN2')
        return tdeltas(delta_x_NN, delta_y_NN, delta_z_NN, delta_e_NN, delta_x_NN2, delta_y_NN2, delta_z_NN2, delta_e_NN2)
    elif twoc == 1:
        tdeltas = namedtuple('tdeltas',
            'delta_x_NN, delta_y_NN, delta_z_NN, delta_x_NN2, delta_y_NN2, delta_z_NN2')
        return tdeltas(delta_x_NN, delta_y_NN, delta_z_NN, delta_x_NN2, delta_y_NN2, delta_z_NN2)
    else:
        tdeltas = namedtuple('tdeltas',
            'delta_x_NN, delta_y_NN, delta_z_NN')
        return tdeltas(delta_x_NN, delta_y_NN, delta_z_NN)
