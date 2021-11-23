# Nicola Dinsdale 2020
# Class to control finetuning
########################################################################################################################
import torch
from torch.autograd import Variable
import numpy as np
from pruning_tools.prune_unet import prune_conv_layer
from operator import itemgetter
from heapq import nsmallest
import json

########################################################################################################################
class FilterPrunner():
    def __init__(self, unet, segmenter, mode='Taylor', use_cuda=False):
        self.unet = unet
        self.segmenter = segmenter
        self.mode = mode
        self.use_cuda = use_cuda
        self.return_ranks = None
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_block = {}

        enc1 = None
        enc2 = None
        enc3 = None
        enc4 = None

        activation_index = 0
        for block_index, block in enumerate(self.unet._modules.items()):
            if not isinstance(block[1], torch.nn.ConvTranspose3d) and not isinstance(block[1], torch.nn.MaxPool3d):
                for layer_index, module in enumerate(block[1]):
                    x = module(x)
                    if self.mode == 'Taylor':
                        if isinstance(module, torch.nn.Conv3d):
                            x.register_hook(self.compute_rank_taylor)
                            self.activations.append(x)
                            self.activation_to_block[activation_index] = [block_index, layer_index]
                            activation_index += 1
                    elif self.mode == 'Random':
                        if isinstance(module, torch.nn.Conv3d):
                            x.register_hook(self.compute_rank_random)
                            self.activations.append(x)
                            self.activation_to_block[activation_index] = [block_index, layer_index]
                            activation_index += 1
                    elif self.mode == 'L1':
                        if isinstance(module, torch.nn.Conv3d):
                            x.register_hook(self.compute_rank_l1)
                            self.activations.append(x)
                            self.activation_to_block[activation_index] = [block_index, layer_index]
                            activation_index += 1
                    elif self.mode == 'L2':
                        if isinstance(module, torch.nn.Conv3d):
                            x.register_hook(self.compute_rank_l2)
                            self.activations.append(x)
                            self.activation_to_block[activation_index] = [block_index, layer_index]
                            activation_index += 1
                    elif self.mode == 'L1_std':
                        if isinstance(module, torch.nn.Conv3d):
                            x.register_hook(self.compute_rank_l1_std)
                            self.activations.append(x)
                            self.activation_to_block[activation_index] = [block_index, layer_index]
                            activation_index += 1
                    else:
                        raise Exception('Unknown mode. Implemented modes: Taylor, Random, L1, L2')
                if block_index == 0:
                    enc1 = x
                if block_index == 2:
                    enc2 = x
                if block_index == 4:
                    enc3 = x
                if block_index == 6:
                    enc4 = x
            else:
                # Pass through the max pooling and upsampling layers layers
                x = block[1](x)
                if block_index == 9:
                    x = torch.cat((x, enc4), dim=1)
                if block_index == 11:
                    x = torch.cat((x, enc3), dim=1)
                if block_index == 13:
                    x = torch.cat((x, enc2), dim=1)
                if block_index == 15:
                    x = torch.cat((x, enc1), dim=1)
        return x

    def compute_rank_taylor(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter,
        # across all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3, 4)).data

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()

            if self.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += taylor.cpu()
        self.grad_index += 1

    def compute_rank_l1(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        l1 = torch.norm(activation, p=1, dim=2)
        l1 = torch.norm(l1, p=1, dim=2)
        l1 = torch.norm(l1, p=1, dim=2)
        l1 = l1.mean(dim=0).data

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()

            if self.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += l1.cpu()
        self.grad_index += 1

    def compute_rank_l1_std(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        l1 = torch.norm(activation, p=1, dim=2)
        l1 = torch.norm(l1, p=1, dim=2)
        l1 = torch.norm(l1, p=1, dim=2)
        l1 = l1.std(dim=0).data

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()

            if self.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += l1.cpu()
        self.grad_index += 1

    def compute_rank_l2(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        l2 = torch.norm(activation, p=2, dim=2)
        l2 = torch.norm(l2, p=2, dim=2)
        l2 = torch.norm(l2, p=2, dim=2)
        l2 = l2.mean(dim=0).data

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()

            if self.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += l2.cpu()
        self.grad_index += 1

    def compute_rank_random(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        if activation.size()[1] != 1:
            taylor = activation * grad
            taylor = taylor.mean(dim=(0, 2, 3, 4)).data

            random = torch.rand(taylor.size())
        else:
            taylor = activation * grad
            taylor = taylor.mean(dim=(0, 2, 3, 4)).data
            random = torch.ones(taylor.size()) * 100

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()

            if self.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += random.cpu()
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        print('number to prune', num)
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_block[i], j, self.filter_ranks[i][j]))
        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        if not self.mode == 'Random':
            for i in self.filter_ranks:
                v = torch.abs(self.filter_ranks[i])
                v = v / np.sqrt(torch.sum(v * v))
                self.filter_ranks[i] = v.cpu()
        print(self.filter_ranks)
        self.return_ranks = self.filter_ranks

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        # After each of the k filters are pruned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_batch = {}
        print('Filters to prune')
        print(filters_to_prune)
        pth = 'pruned_filter_path'
        file = open(pth, "a+")

        for ([b, l], f, _) in filters_to_prune:
            print(b, l, f)
            file.write(str(b) + '\t' + str(l) + '\t' + str(f) +'\t')
            file.write('\n')

            if b not in filters_to_prune_per_batch:
                filters_to_prune_per_batch[b] = {l : []}

            if l not in filters_to_prune_per_batch[b].keys():
                filters_to_prune_per_batch[b][l] = []
            filters_to_prune_per_batch[b][l].append(f)

        for b in filters_to_prune_per_batch:
            for l in filters_to_prune_per_batch[b]:
                filters_to_prune_per_batch[b][l] = sorted(filters_to_prune_per_batch[b][l])
                for i in range(len(filters_to_prune_per_batch[b][l])):
                    filters_to_prune_per_batch[b][l][i] = filters_to_prune_per_batch[b][l][i] - i

        filters_to_prune = []
        for b in filters_to_prune_per_batch:
            for l in filters_to_prune_per_batch[b]:
                for i in filters_to_prune_per_batch[b][l]:
                    filters_to_prune.append((b, l, i))

        return filters_to_prune


class PruningController():
    def __init__(self, unet, segmenter, train_data_loader, criterion, prune_percentage = 10, mode = 'Taylor', pruning_pth=None, use_cuda=False):
        self.unet = unet
        self.segmenter = segmenter
        self.train_data_loader = train_data_loader
        self.mode = mode
        self.pruner = FilterPrunner(self.unet, self.segmenter, self.mode)
        self.criterion = criterion
        self.use_cuda = use_cuda
        self.mode = mode
        self.prune_percentage = prune_percentage
        self.pruning_pth = pruning_pth

    def train_batch(self, optimizer, batch, label, rank_filters):
        if self.use_cuda:
            batch = batch.cuda()
            label = label.cuda()

        self.unet.zero_grad()
        self.segmenter.zero_grad()
        input = Variable(batch)

        if rank_filters:
            embeddings = self.pruner.forward(input)
            output = self.segmenter(embeddings)
            self.criterion(output, Variable(label)).backward()

        else:
            embeddings = self.unet(input)
            self.criterion(self.segmenter(embeddings), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer=None, rank_filters=False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.pruner.reset()
        self.train_epoch(rank_filters = True)
        self.pruner.normalize_ranks_per_layer()
        returned_ranks = self.pruner.return_ranks
        return self.pruner.get_prunning_plan(num_filters_to_prune), returned_ranks

    def total_num_filters(self):
        filters = 0
        for name, block in self.unet._modules.items():
            if not isinstance(block, torch.nn.ConvTranspose3d) and not isinstance(block, torch.nn.MaxPool3d):
                for module in block:
                    if isinstance(module, torch.nn.Conv3d):
                        filters = filters + module.out_channels
        return filters

    def prune(self):
        # Make sure all the parameters are trainable
        for param in self.unet.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = int(self.prune_percentage)
        print('Number of filters to prune', num_filters_to_prune_per_iteration)

        print('Getting Candidates', flush=True)
        prune_targets, returned_ranks = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)

        layers_pruned = {}

        for block_index, layer_index, filter_index in prune_targets:
            if block_index not in layers_pruned:
                layers_pruned[block_index] = {layer_index: 0}
            if layer_index not in layers_pruned[block_index]:
                layers_pruned[block_index][layer_index] = 0
            layers_pruned[block_index][layer_index] = layers_pruned[block_index][layer_index] + 1

        print("Layers that will be pruned", layers_pruned, flush=True)
        if self.pruning_pth:
            file_object = open(self.pruning_pth, 'a')
            file_object.write('\n')
            layers_pruned_str = json.dumps(layers_pruned)
            file_object.write(layers_pruned_str)
            file_object.close()

        print("Pruning filters")

        unet = self.unet.cpu()
        segmenter = self.segmenter.cpu()
        for batch_index, layer_index, filter_index in prune_targets:
            unet, segmenter = prune_conv_layer(unet, segmenter, batch_index, layer_index, filter_index, use_cuda=self.use_cuda)

        self.unet = unet
        self.segmenter = segmenter
        if self.use_cuda:
            self.unet = self.unet.cuda()
            self.segmenter = self.segmenter.cuda()

        message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
        print("Filters pruned", str(message))

        return returned_ranks


