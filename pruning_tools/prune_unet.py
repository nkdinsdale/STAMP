# Nicola Dinsdale 2020
# Create a tool for model pruning
########################################################################################################################
import torch
import numpy as np
import time
########################################################################################################################

def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]

def replace_layer_new(model,  block_index, layer_index, layer):
    if not isinstance(layer, torch.nn.ConvTranspose3d):
        block_key = list(model._modules.keys())[block_index]
        model._modules[block_key][layer_index] = layer
    else:
        block_key = list(model._modules.keys())[block_index]
        model._modules[block_key] = layer

    return model

def replace_layer_new2d(model,  block_index, layer_index, layer):
    if not isinstance(layer, torch.nn.ConvTranspose2d):
        block_key = list(model._modules.keys())[block_index]
        model._modules[block_key][layer_index] = layer
    else:
        block_key = list(model._modules.keys())[block_index]
        model._modules[block_key] = layer

    return model

def prune_conv_layer(model, model_2,  block_index, layer_index, filter_index, use_cuda=False):
    conv = list(model._modules.items())[block_index][1][layer_index]
    next_conv = None
    block_offset = 0
    new_index = layer_index + 1
    flag = 0

    # Find the next conv after the current one
    # The next conv could be a transpose conv
    while block_index + block_offset < len(model._modules.items()):
        next_block = list(model._modules.items())[block_index + block_offset][1]
        if not isinstance(next_block, torch.nn.MaxPool3d) and not isinstance(next_block, torch.nn.ConvTranspose3d):
            while new_index < len(next_block):
                res = next_block[new_index]
                if isinstance(res, torch.nn.Conv3d):
                    next_conv = res
                    flag += 1
                    break
                new_index = new_index + 1
        elif isinstance(next_block, torch.nn.ConvTranspose3d):
            res = next_block
            next_conv = res
            flag += 1
            break
        if flag == 1:
            break

        new_index = 0
        block_offset = block_offset + 1

    new_block = block_index + block_offset

    # Define the new convolution out channels is decreased by one because we have pruned one of the filters
    new_conv = torch.nn.Conv3d(in_channels = conv.in_channels,
                               out_channels = conv.out_channels - 1,
                               kernel_size = conv.kernel_size,
                               stride = conv.stride,
                               padding = conv.padding,
                               dilation = conv.dilation,
                               groups = conv.groups,
                               bias = (conv.bias is not None))

    # Copy all the weights but the ones for the filter we have pruned
    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    new_weights[:filter_index, :, :, :, :] = old_weights[:filter_index, :, :, :, :]
    new_weights[filter_index:, :, :, :, :] = old_weights[filter_index + 1: :, :, :, :]


    # Add the weights for the new model
    new_conv.weight.data = torch.from_numpy(new_weights)
    if use_cuda:
        new_conv.weight.data = new_conv.weight.data.cuda()
    # Update the biases
    bias_numpy = conv.bias.data.cpu().numpy()
    bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype=np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index :] = bias_numpy[filter_index+1 :]
    new_conv.bias.data = torch.from_numpy(bias)

    if use_cuda:
        new_conv.bias.data = new_conv.bias.data.cuda()

    # Update the batchnorm if they are in the model
    batch_offset = 2       # Conv relu batch
    batch_norm = None
    while layer_index + batch_offset < len(list(model._modules.items())[block_index][1]):
        res = list(model._modules.items())[block_index][1][layer_index + batch_offset]
        if isinstance(res, torch.nn.BatchNorm3d):
            batch_norm = res
            break
    new_batch = None
    if not batch_norm is None:
        new_batch = torch.nn.BatchNorm3d(num_features=batch_norm.num_features - 1)
        old_weights = batch_norm.weight.data.cpu().numpy()
        new_weights = new_batch.weight.data.cpu().numpy()
        new_weights[:filter_index] = old_weights[:filter_index]
        new_weights[filter_index:] = old_weights[filter_index + 1 :]

        # Add the weights for the new model
        new_batch.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            new_batch.weight.data = new_batch.weight.data.cuda()

    if not next_conv is None:
        if isinstance(next_conv, torch.nn.Conv3d):
            next_new_conv = torch.nn.Conv3d(in_channels = next_conv.in_channels -1,
                                            out_channels = next_conv.out_channels,
                                            kernel_size = next_conv.kernel_size,
                                            stride = next_conv.stride,
                                            padding = next_conv.padding,
                                            dilation = next_conv.dilation,
                                            groups = next_conv.groups,
                                            bias = (next_conv.bias is not None))
            old_weights = next_conv.weight.data.cpu().numpy()
            new_weights = next_new_conv.weight.data.cpu().numpy()

            new_weights[:, : filter_index, :, :, :] = old_weights[:, : filter_index, :, :, :]
            new_weights[:, filter_index:, :, :, :] = old_weights[:, filter_index + 1:, :, :, :]

        elif isinstance(next_conv, torch.nn.ConvTranspose3d):
            next_new_conv = torch.nn.ConvTranspose3d(in_channels=next_conv.in_channels - 1,
                                                     out_channels=next_conv.out_channels,
                                                     kernel_size=next_conv.kernel_size,
                                                     stride=next_conv.stride,
                                                     padding=next_conv.padding,
                                                     groups=next_conv.groups,
                                                     bias=(next_conv.bias is not None))
            old_weights = next_conv.weight.data.cpu().numpy()
            new_weights = next_new_conv.weight.data.cpu().numpy()

            new_weights[ : filter_index, ] = old_weights[: filter_index ]
            new_weights[filter_index:] = old_weights[filter_index + 1:]

        # Add the weights for the new model
        next_new_conv.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            next_new_conv.weight.data = next_new_conv.weight.data.cuda()

        next_new_conv.bias.data = next_conv.bias.data

    # Also need to prune the corresponding transposed dec layer where they concatenate the layers
    # Depends on what the batch index is that we pruned from and the layer index, only matters if it was the last conv
    # layer in that block so if new block is not block we know this matters
    new_dec_conv = None
    dec_index = None
    dec_offset = None
    dec_conv = None

    if block_index != new_block:
        if block_index in [0, 2, 4, 6]:
            if block_index == 0:
                dec_index = 16
            if block_index == 2:
                dec_index = 14
            if block_index == 4:
                dec_index = 12
            if block_index == 6:
                dec_index = 10
        if dec_index:
            dec_offset = 0
            while layer_index + dec_offset < len(list(model._modules.items())[dec_index][0]):
                res = list(model._modules.items())[dec_index][1][dec_offset]
                if isinstance(res, torch.nn.Conv3d):
                    dec_conv = res
                    break
                dec_offset += 1

            new_dec_conv = torch.nn.Conv3d(in_channels = dec_conv.in_channels -1,
                                            out_channels = dec_conv.out_channels,
                                            kernel_size = dec_conv.kernel_size,
                                            stride = dec_conv.stride,
                                            padding = dec_conv.padding,
                                            dilation = dec_conv.dilation,
                                            groups = dec_conv.groups,
                                            bias = (dec_conv.bias is not None))
            old_weights = dec_conv.weight.data.cpu().numpy()
            new_weights = new_dec_conv.weight.data.cpu().numpy()

            # Will be pruning from the second half I think
            filters = int(dec_conv.in_channels/2)
            new_weights[:, :  filters +filter_index, :, :, :] = old_weights[:, : filters + filter_index, :, :, :]
            new_weights[:, filters + filter_index:, :, :, :] = old_weights[:, filters + filter_index + 1:, :, :, :]

            # Add the weights for the new model
            new_dec_conv.weight.data = torch.from_numpy(new_weights)
            if use_cuda:
                new_dec_conv.weight.data = new_dec_conv.weight.data.cuda()

            # Update the bias
            new_dec_conv.bias.data = dec_conv.bias.data

    if not next_conv is None:
         # First update the conv layer, its address is block_index, layer_index
        model = replace_layer_new(model, block_index, layer_index, new_conv)
        # Now replace the batchnorm, block_index, filter_index + batch_offset
        if new_batch:
            model = replace_layer_new(model, block_index, layer_index+batch_offset, new_batch)
        # Now replace the next new conv
        model = replace_layer_new(model, block_index+block_offset, new_index, next_new_conv)
        # If there is a cat depending on it update that too
        if new_dec_conv:
            model = replace_layer_new(model, dec_index, dec_offset, new_dec_conv)

    if next_conv is None:
        # Next conv is in model 2
        model_2_index = 0
        model_2_layer_index = 0
        while model_2_layer_index < len(list(model_2._modules.items())[model_2_index]):
            res = list(model_2._modules.items())[model_2_index][1][model_2_layer_index]
            if isinstance(res, torch.nn.Conv3d):
                next_conv = res
                break
            model_2_layer_index += 1

        next_new_conv = torch.nn.Conv3d(in_channels=next_conv.in_channels - 1,
                                        out_channels=next_conv.out_channels,
                                        kernel_size=next_conv.kernel_size,
                                        stride=next_conv.stride,
                                        padding=next_conv.padding,
                                        dilation=next_conv.dilation,
                                        groups=next_conv.groups,
                                        bias=(next_conv.bias is not None))
        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :, :] = old_weights[:, : filter_index, :, :, :]
        new_weights[:, filter_index:, :, :, :] = old_weights[:, filter_index + 1:, :, :, :]

        # Add the weights for the new model
        next_new_conv.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            next_new_conv.weight.data = next_new_conv.weight.data.cuda()

        next_new_conv.bias.data = next_conv.bias.data

        # Now update the bits of the model
         # First update the conv layer, its address is block_index, layer_index
        model = replace_layer_new(model, block_index, layer_index, new_conv)
        # Now replace the batchnorm, block_index, filter_index + batch_offset
        if new_batch:
            model = replace_layer_new(model, block_index, layer_index+batch_offset, new_batch)
        # Now replace the next new conv, this is in model 2 not model 1
        model_2 = replace_layer_new(model_2, model_2_index, model_2_layer_index, next_new_conv)

    return model, model_2


def prune_conv_layer2d(model, model_2,  block_index, layer_index, filter_index, use_cuda=False):
    conv = list(model._modules.items())[block_index][1][layer_index]
    next_conv = None
    block_offset = 0
    new_index = layer_index + 1
    flag = 0

    # Find the next conv after the current one
    # The next conv could be a transpose conv
    print('beginning pruning')
    while block_index + block_offset < len(model._modules.items()):
        next_block = list(model._modules.items())[block_index + block_offset][1]
        if not isinstance(next_block, torch.nn.MaxPool2d) and not isinstance(next_block, torch.nn.ConvTranspose2d):
            while new_index < len(next_block):
                res = next_block[new_index]
                if isinstance(res, torch.nn.Conv2d):
                    next_conv = res
                    flag += 1
                    break
                new_index = new_index + 1
        elif isinstance(next_block, torch.nn.ConvTranspose2d):
            res = next_block
            next_conv = res
            flag += 1
            break
        if flag == 1:
            break

        new_index = 0
        block_offset = block_offset + 1

    new_block = block_index + block_offset
    # Define the new convolution out channels is decreased by one because we have pruned one of the filters
    new_conv = torch.nn.Conv2d(in_channels = conv.in_channels,
                               out_channels = conv.out_channels - 1,
                               kernel_size = conv.kernel_size,
                               stride = conv.stride,
                               padding = conv.padding,
                               dilation = conv.dilation,
                               groups = conv.groups,
                               bias = (conv.bias is not None))

    # Copy all the weights but the ones for the filter we have pruned
    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    new_weights[:filter_index, :, :, :] = old_weights[:filter_index, :, :, :]
    new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1: :, :, :]


    # Add the weights for the new model
    new_conv.weight.data = torch.from_numpy(new_weights)
    if use_cuda:
        new_conv.weight.data = new_conv.weight.data.cuda()
    # Update the biases
    bias_numpy = conv.bias.data.cpu().numpy()
    bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype=np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index :] = bias_numpy[filter_index+1 :]
    new_conv.bias.data = torch.from_numpy(bias)

    if use_cuda:
        new_conv.bias.data = new_conv.bias.data.cuda()

    # Update the batchnorm if they are in the model
    batch_offset = 2       # Conv relu batch
    batch_norm = None
    while layer_index + batch_offset < len(list(model._modules.items())[block_index][1]):
        res = list(model._modules.items())[block_index][1][layer_index + batch_offset]
        if isinstance(res, torch.nn.BatchNorm2d):
            batch_norm = res
            break

    new_batch = None
    if not batch_norm is None:
        new_batch = torch.nn.BatchNorm2d(num_features=batch_norm.num_features - 1)
        old_weights = batch_norm.weight.data.cpu().numpy()
        new_weights = new_batch.weight.data.cpu().numpy()
        new_weights[:filter_index] = old_weights[:filter_index]
        new_weights[filter_index:] = old_weights[filter_index + 1 :]

        # Add the weights for the new model
        new_batch.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            new_batch.weight.data = new_batch.weight.data.cuda()

    if not next_conv is None:
        if isinstance(next_conv, torch.nn.Conv2d):
            next_new_conv = torch.nn.Conv2d(in_channels = next_conv.in_channels -1,
                                            out_channels = next_conv.out_channels,
                                            kernel_size = next_conv.kernel_size,
                                            stride = next_conv.stride,
                                            padding = next_conv.padding,
                                            dilation = next_conv.dilation,
                                            groups = next_conv.groups,
                                            bias = (next_conv.bias is not None))
            old_weights = next_conv.weight.data.cpu().numpy()
            new_weights = next_new_conv.weight.data.cpu().numpy()

            new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
            new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]

        elif isinstance(next_conv, torch.nn.ConvTranspose2d):
            next_new_conv = torch.nn.ConvTranspose2d(in_channels=next_conv.in_channels - 1,
                                                     out_channels=next_conv.out_channels,
                                                     kernel_size=next_conv.kernel_size,
                                                     stride=next_conv.stride,
                                                     padding=next_conv.padding,
                                                     groups=next_conv.groups,
                                                     bias=(next_conv.bias is not None))
            old_weights = next_conv.weight.data.cpu().numpy()
            new_weights = next_new_conv.weight.data.cpu().numpy()

            new_weights[ : filter_index, ] = old_weights[: filter_index ]
            new_weights[filter_index:] = old_weights[filter_index + 1:]

        # Add the weights for the new model
        next_new_conv.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            next_new_conv.weight.data = next_new_conv.weight.data.cuda()

        next_new_conv.bias.data = next_conv.bias.data

    # Also need to prune the corresponding transposed dec layer where they concatenate the layers
    # Depends on what the batch index is that we pruned from and the layer index, only matters if it was the last conv
    # layer in that block so if new block is not block we know this matters
    new_dec_conv = None
    dec_index = None
    dec_offset = None
    dec_conv = None

    if block_index != new_block:
        if block_index in [0, 2, 4, 6]:
            if block_index == 0:
                dec_index = 16
            if block_index == 2:
                dec_index = 14
            if block_index == 4:
                dec_index = 12
            if block_index == 6:
                dec_index = 10
        if dec_index:
            dec_offset = 0
            while layer_index + dec_offset < len(list(model._modules.items())[dec_index][0]):
                res = list(model._modules.items())[dec_index][1][dec_offset]
                if isinstance(res, torch.nn.Conv2d):
                    dec_conv = res
                    break
                dec_offset += 1

            new_dec_conv = torch.nn.Conv2d(in_channels = dec_conv.in_channels -1,
                                            out_channels = dec_conv.out_channels,
                                            kernel_size = dec_conv.kernel_size,
                                            stride = dec_conv.stride,
                                            padding = dec_conv.padding,
                                            dilation = dec_conv.dilation,
                                            groups = dec_conv.groups,
                                            bias = (dec_conv.bias is not None))
            old_weights = dec_conv.weight.data.cpu().numpy()
            new_weights = new_dec_conv.weight.data.cpu().numpy()

            # Will be pruning from the second half I think
            filters = int(dec_conv.in_channels/2)
            new_weights[:, :  filters +filter_index, :, :] = old_weights[:, : filters + filter_index, :, :]
            new_weights[:, filters + filter_index:, :, :] = old_weights[:, filters + filter_index + 1:, :, :]

            # Add the weights for the new model
            new_dec_conv.weight.data = torch.from_numpy(new_weights)
            if use_cuda:
                new_dec_conv.weight.data = new_dec_conv.weight.data.cuda()

            # Update the bias
            new_dec_conv.bias.data = dec_conv.bias.data

    if not next_conv is None:
         # First update the conv layer, its address is block_index, layer_index
        model = replace_layer_new2d(model, block_index, layer_index, new_conv)
        # Now replace the batchnorm, block_index, filter_index + batch_offset
        if new_batch:
            model = replace_layer_new2d(model, block_index, layer_index+batch_offset, new_batch)
        # Now replace the next new conv
        model = replace_layer_new2d(model, block_index+block_offset, new_index, next_new_conv)
        # If there is a cat depending on it update that too
        if new_dec_conv:
            model = replace_layer_new2d(model, dec_index, dec_offset, new_dec_conv)

    if next_conv is None:
        # Next conv is in model 2
        model_2_index = 0
        model_2_layer_index = 0
        while model_2_layer_index < len(list(model_2._modules.items())[model_2_index]):
            res = list(model_2._modules.items())[model_2_index][1][model_2_layer_index]
            if isinstance(res, torch.nn.Conv2d):
                next_conv = res
                break
            model_2_layer_index += 1

        next_new_conv = torch.nn.Conv2d(in_channels=next_conv.in_channels - 1,
                                        out_channels=next_conv.out_channels,
                                        kernel_size=next_conv.kernel_size,
                                        stride=next_conv.stride,
                                        padding=next_conv.padding,
                                        dilation=next_conv.dilation,
                                        groups=next_conv.groups,
                                        bias=(next_conv.bias is not None))
        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]

        # Add the weights for the new model
        next_new_conv.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            next_new_conv.weight.data = next_new_conv.weight.data.cuda()

        next_new_conv.bias.data = next_conv.bias.data

        # Now update the bits of the model
         # First update the conv layer, its address is block_index, layer_index
        model = replace_layer_new2d(model, block_index, layer_index, new_conv)
        # Now replace the batchnorm, block_index, filter_index + batch_offset
        if new_batch:
            model = replace_layer_new2d(model, block_index, layer_index+batch_offset, new_batch)
        # Now replace the next new conv, this is in model 2 not model 1
        model_2 = replace_layer_new2d(model_2, model_2_index, model_2_layer_index, next_new_conv)

    return model, model_2

