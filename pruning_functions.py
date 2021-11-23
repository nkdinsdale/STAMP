# Nicola Dinsdale 2020
# Make files to do the pruning in a loop so we can automate the process
########################################################################################################################
import torch
from losses.dice_loss import dice_loss
import numpy as np
import torch.optim as optim
import sys
from pruning_tools.prune_manager_unet_fixed_filters import PruningController
from torch.autograd import Variable

########################################################################################################################
def train_normal(args, models, train_loader, optimizer, criterion, epoch):
    cuda = torch.cuda.is_available()

    [encoder, regressor] = models

    total_loss = 0

    encoder.train()
    regressor.train()

    batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        if list(data.size())[0] == args.batch_size:
            batches += 1

            # First update the encoder and regressor
            optimizer.zero_grad()
            features = encoder(data)
            x = regressor(features)
            loss = criterion(x, target)
            loss.backward()
            optimizer.step()

            total_loss += loss

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.item()), flush=True)
            del loss
            del features

    av_loss = total_loss / batches
    av_loss_copy = np.copy(av_loss.detach().cpu().numpy())

    del av_loss

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss_copy, flush=True))

    return av_loss_copy, np.NaN


def val_normal(args, models, val_loader, criterion):
    cuda = torch.cuda.is_available()

    [encoder, regressor] = models

    encoder.eval()
    regressor.eval()

    total_loss = 0

    batches = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            batches += 1
            features = encoder(data)
            x = regressor(features)

            loss = criterion(x, target)

            total_loss += loss

    av_loss = total_loss / batches
    av_loss_copy = np.copy(av_loss.detach().cpu().numpy())
    del av_loss

    print('Validation set: Average Domain loss: {:.4f}\n'.format(av_loss_copy, flush=True))

    return av_loss_copy, np.NaN


def pruning_function(iteration, models, pths, train_dataloader, pruning_percentage, pruning_mode):
    [unet, segmenter] = models
    [unet_save_pth, segmenter_save_pth, pruning_pth, _] = pths

    print('Pruning Iteration: ', iteration)

    criterion = dice_loss()
    criterion.cuda()

    fine_tuner = PruningController(unet, segmenter, train_dataloader, criterion, prune_percentage=pruning_percentage, mode=pruning_mode, pruning_pth=pruning_pth, use_cuda=True)
    returned_ranks = fine_tuner.prune()
    new_drop = new_weights_rankval(returned_ranks)
    print('New Dropout Values = ', new_drop)
    unet = update_droplayer(unet, new_drop)

    torch.save(unet, unet_save_pth+'_pruned_iter_'+str(iteration))
    torch.save(segmenter, segmenter_save_pth+'_pruned_iter_'+str(iteration))

    return unet, segmenter

def new_weights(returned_ranks):
    for key in returned_ranks:
        tot = len(returned_ranks[key])
        returned_ranks[key] = (torch.sum(returned_ranks[key]) / tot).numpy()
        returned_ranks[key] = 1 - returned_ranks[key]

    m = max(returned_ranks.values())

    for key in returned_ranks:
        returned_ranks[key] = returned_ranks[key] / m

    base_prob = 0.10
    for key in returned_ranks:
        returned_ranks[key] = returned_ranks[key] * base_prob
    return returned_ranks

def new_weights_rankval(d):
    new_d = {}
    for key in d:
        array = d[key].numpy()
        for i in array:
            if i not in new_d.keys():
                new_d[i] = key
            else:
                i += 1e-12
                new_d[i] = key
    o_keys = sorted(new_d.keys())

    i = len(o_keys)
    prune_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
                  16: 0}
    for val in o_keys:
        i -= 1
        index = new_d[val]
        prune_dict[index] += i

    for key in prune_dict:
        prune_dict[key] = prune_dict[key] / len(d[key].numpy())
    max_val = max(prune_dict.values())
    for key in prune_dict:
        prune_dict[key] = prune_dict[key] / max_val

    base_prob = 0.10
    for key in prune_dict:
        prune_dict[key] = prune_dict[key] * base_prob
    return prune_dict

def update_droplayer(model, new_drop):
    model.encoder1.enc1drop1 = torch.nn.Dropout3d(p=new_drop[0])
    model.encoder1.enc1drop2 = torch.nn.Dropout3d(p=new_drop[1])
    model.encoder2.enc2drop1 = torch.nn.Dropout3d(p=new_drop[2])
    model.encoder2.enc2drop2 = torch.nn.Dropout3d(p=new_drop[3])
    model.encoder3.enc3drop1 = torch.nn.Dropout3d(p=new_drop[4])
    model.encoder3.enc3drop2 = torch.nn.Dropout3d(p=new_drop[5])
    model.encoder4.enc4drop1 = torch.nn.Dropout3d(p=new_drop[6])
    model.encoder4.enc4drop2 = torch.nn.Dropout3d(p=new_drop[7])
    model.bottleneck.bottleneckdrop1 = torch.nn.Dropout3d(p=new_drop[8])
    model.bottleneck.bottleneckdrop2 = torch.nn.Dropout3d(p=new_drop[9])
    model.decoder4.dec4drop1 = torch.nn.Dropout3d(p=new_drop[10])
    model.decoder4.dec4drop2 = torch.nn.Dropout3d(p=new_drop[11])
    model.decoder3.dec3drop1 = torch.nn.Dropout3d(p=new_drop[12])
    model.decoder3.dec3drop2 = torch.nn.Dropout3d(p=new_drop[13])
    model.decoder2.dec2drop1 = torch.nn.Dropout3d(p=new_drop[14])
    model.decoder2.dec2drop2 = torch.nn.Dropout3d(p=new_drop[15])
    model.decoder1.dec1drop2 = torch.nn.Dropout3d(p=new_drop[16])
    return model

def tuning_function(args, no_recovery_epochs, loss_store, iteration, models, pths, train_dataloader, val_dataloader, early_stopping):
    [unet, segmenter] = models
    [unet_save_pth, segmenter_save_pth, _, loss_pth] = pths
    print('Pruning Iteration: ', iteration)

    criterion = dice_loss()
    criterion.cuda()

    optimizer = optim.Adam(list(unet.parameters()) + list(segmenter.parameters()), lr=args.learning_rate)

    epoch = 1
    while epoch < no_recovery_epochs:
        print('Epoch ', epoch, '/', args.epochs, flush=True)

        loss, _ = train_normal(args, models, train_dataloader, optimizer, criterion, epoch)
        val_loss, _ = val_normal(args, models, val_dataloader, criterion)
        loss_store.append([loss, val_loss])
        np.save(loss_pth, np.array(loss_store))

        unet_save_pth_chk = unet_save_pth+'_finetuned_iter_'+str(iteration)
        segmenter_save_pth_chk = segmenter_save_pth+'_finetuned_iter_'+str(iteration)

        #Save the pruned models
        torch.save(unet, unet_save_pth_chk)
        torch.save(segmenter, segmenter_save_pth_chk)

        early_stopping(val_loss, models, epoch, optimizer, loss, [unet_save_pth, segmenter_save_pth])

        if early_stopping.early_stop:
            loss_store = np.array(loss_store)
            np.save(loss_pth, loss_store)
            sys.exit('Patience Reached - Early Stopping Activated')

        epoch += 1

        torch.cuda.empty_cache()  # Clear memory cache

    return loss_store