# Nicola Dinsdale 2021
# Main file to control the pruning loop
########################################################################################################################
from pruning_functions import pruning_function, train_normal, val_normal, tuning_function
from models.unet_model_targeted_dropout import UNet, Segmenter
from datasets.numpy_dataset import numpy_dataset
from torch.utils.data import DataLoader
import torch
from losses.dice_loss import dice_loss
import numpy as np
from sklearn.utils import shuffle
from utils import Args, EarlyStopping_split_models_pruning_unet
import torch.optim as optim
import sys
import argparse

########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 1000
args.batch_size = 8
args.patience =  10
args.learning_rate = 1e-2

pruning_freq = 1

torch.cuda.empty_cache()
cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Define Inputs for pruning model')
parser.add_argument('-m', action="store", dest="mode")
parser.add_argument('-p', action="store", dest="percentage")
parser.add_argument('-r', action='store', dest='no_recovery_epochs')
parser.add_argument('-i', action='store', dest='starting_iteration')
parser.add_argument('-no', action='store', dest='no_iterations')

results = parser.parse_args()
try:
    pruning_percentage = int(results.percentage)
    print('Pruning Percentage = ', pruning_percentage)
    pruning_mode = results.mode
    print('Pruning Mode = ', pruning_mode)
    no_recovery_epochs = int(results.no_recovery_epochs)
    print('No. Recovery Epochs =', int(no_recovery_epochs))
    starting_iteration = int(results.starting_iteration)
    print('Starting Iteration =', int(starting_iteration))
    no_iterations = int(results.no_iterations)
    print('Number of interations =', no_iterations)
except:
    raise Exception('Argument not supplied')


iteration = starting_iteration
if iteration == 0:
    LOAD_PATH_UNET = None
    LOAD_PATH_SEGMENTER = None
else:
    LOAD_PATH_UNET = 'pth' + str(iteration)
    LOAD_PATH_SEGMENTER = 'pth' + str(iteration)

PATH_UNET = 'pth_unet'
PATH_SEGMENTER = 'pth_segmenter'
CHK_PATH_UNET = 'pth_unet_chk'
CHK_PATH_SEGMENTER = 'pth_segmenter_chk'
PRUNING_PATH = 'pruning_pth'
LOSS_PATH = 'loss_pth'


########################################################################################################################
X = np.load('X_train_pths.npy')
y = np.load('y_train_pths.npy')

X, y = shuffle(X, y, random_state=0)  # Same seed everytime
proportion = int(args.train_val_prop * len(X))
X_train = X[:proportion]
X_val = X[proportion:]
y_train = y[:proportion]
y_val = y[proportion:]

print('Data splits')
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

print('Creating datasets and dataloaders')
train_dataset = numpy_dataset(X_train, y_train)
val_dataset = numpy_dataset(X_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

features = 4

probs = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1, 10: 0.1, 11: 0.1, 12: 0.1, 13: 0.1, 14: 0.1, 15: 0.1, 16: 0.1}
base_prob = 0.05
for key in probs:
    probs[key] = probs[key] * base_prob

if not LOAD_PATH_UNET:
    unet = UNet(drop_probs=probs, init_features=int(features))
    segmenter = Segmenter(out_channels=2, init_features=int(features))
else:
    unet = torch.load(LOAD_PATH_UNET, map_location=lambda storage, loc: storage)
    segmenter = torch.load(LOAD_PATH_SEGMENTER, map_location=lambda storage, loc: storage)
if cuda:
    unet = unet.cuda()
    segmenter = segmenter.cuda()


criterion = dice_loss()
criterion.cuda()

optimizer = optim.Adam(list(unet.parameters()) + list(segmenter.parameters()), lr=args.learning_rate)

# Initalise the early stopping
early_stopping = EarlyStopping_split_models_pruning_unet(args.patience, verbose=False)

epoch_reached = 1
loss_store = []

models = [unet, segmenter]

epoch = 2
for iteration in range(iteration, iteration + no_iterations):
    if cuda:
        unet = unet.cuda()
        segmenter = segmenter.cuda()

    if iteration == 0:
        for epoch in range(1, 2):
            print('Epoch ', epoch, '/', args.epochs, flush=True)
            loss, _ = train_normal(args, models, train_dataloader, optimizer, criterion, epoch)

            val_loss, _ = val_normal(args, models, val_dataloader, criterion)
            loss_store.append([loss, val_loss])
            np.save(LOSS_PATH, np.array(loss_store))

            # Decide whether the model should stop training or not
            early_stopping(val_loss, models, epoch, optimizer, loss, [CHK_PATH_UNET, CHK_PATH_SEGMENTER])

            if early_stopping.early_stop:
                loss_store = np.array(loss_store)
                np.save(LOSS_PATH, loss_store)
                sys.exit('Patience Reached - Early Stopping Activated')

            if epoch == args.epochs:
                print('Finished Training', flush=True)
                print('Saving the model', flush=True)

                # Save the model in such a way that we can continue training later
                torch.save(unet, PATH_UNET)
                torch.save(segmenter, PATH_SEGMENTER)

                loss_store = np.array(loss_store)
                np.save(LOSS_PATH, loss_store)

    if epoch % pruning_freq == 0:
        iteration += 1
        pths = [PATH_UNET, PATH_SEGMENTER, PRUNING_PATH, LOSS_PATH]
        unet, segmenter = pruning_function(iteration, models, pths, train_dataloader, pruning_percentage, pruning_mode)

        unet = torch.load(PATH_UNET+'_pruned_iter_'+str(iteration), map_location=lambda storage, loc: storage)
        segmenter = torch.load(PATH_SEGMENTER+'_pruned_iter_'+str(iteration), map_location=lambda storage, loc: storage)

        unet = unet.cuda()
        segmenter = segmenter.cuda()

        models = [unet, segmenter]

        loss_store = tuning_function(args, no_recovery_epochs, loss_store, iteration, models, pths, train_dataloader, val_dataloader, early_stopping)

    torch.cuda.empty_cache()  # Clear memory cache



