import sys
import torch
import click
import json
import datetime
import os
from timeit import default_timer as timer

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
import types
import pdb

from tqdm import tqdm
from tensorboardX import SummaryWriter

import losses
import datasets
import metrics
import model_selector

NUM_EPOCHS = 100

@click.command()
@click.option('--param_file', default='params_celeba.json', help='JSON parameters file')
def train_multi_task(param_file):
    with open('configs.json') as config_params:
        configs = json.load(config_params)

    with open(param_file) as json_params:
        params = json.load(json_params)


    exp_identifier = []
    for (key, val) in params.items():
        if key != "optimizer" and key != "batch_size" and key != "lr" and key != "dataset" and key != "rw" and key != "norm_type":
            continue
        exp_identifier+= ['{}={}'.format(key,val)]

    exp_identifier = '_'.join(exp_identifier)
    params['exp_id'] = exp_identifier
    
    device = torch.device('cuda:0')
    writer = SummaryWriter(log_dir='runs/{}_{}'.format(params['exp_id'], datetime.datetime.now().strftime("%I_%M%p on %B %d, %Y")))

    train_loader, train_dst, val_loader, val_dst, test_loader, test_dst = datasets.get_dataset(params, configs, device)
    loss_fn = losses.get_loss(params)
    metric_main = metrics.get_metrics(params)
    metric_base = metrics.get_metrics(params)

    model_main = model_selector.get_model(params, device)
    model_base = model_selector.get_model(params, device)
    model_main_params = []
    model_base_params = []
    for m in model_main:
        model_main_params += model_main[m].parameters()
        model_base_params += model_base[m].parameters()
    if 'RMSprop' in params['optimizer']:
        optimizer_main = torch.optim.RMSprop(model_main_params, lr=params['lr'])
        optimizer_base = torch.optim.RMSprop(model_base_params, lr=params['lr'])
    elif 'Adam' in params['optimizer']:
        optimizer_main = torch.optim.Adam(model_main_params, lr=params['lr'])
        optimizer_base = torch.optim.Adam(model_base_params, lr=params['lr'])
    elif 'SGD' in params['optimizer']:
        optimizer_main = torch.optim.SGD(model_main_params, lr=params['lr'], momentum=0.9)
        optimizer_base = torch.optim.SGD(model_base_params, lr=params['lr'], momentum=0.9)

    tasks = params['tasks']
    task_num = len(tasks)
    rw = params['rw']
    C = params['C']
    if params["norm_type"] == "spectral":
        ord = 2
    else:
        ord = "fro"
    all_tasks = configs[params['dataset']]['all_tasks']
    print('Starting training with parameters \n \t{} \n'.format(str(params)))

    n_iter = 0
    loss_init = {}
    for epoch in tqdm(range(NUM_EPOCHS)):
        start = timer()
        print('Epoch {} Started'.format(epoch))
        if (epoch+1) % 10 == 0:
            # Every 30 epoch, half the LR
            for param_group in optimizer_main.param_groups:
                param_group['lr'] *= 0.8
            for param_group in optimizer_base.param_groups:
                param_group['lr'] *= 0.8
            print('Half the learning rate{}'.format(n_iter))

        for m in model_main:
            model_main[m].train()
            model_base[m].train()
        
        for batch in train_loader:
            n_iter += 1
            count = 0
            # First member is always images
            images = batch[0].to(device)
            images.requires_grad = True

            labels = {}
            # Read all targets of all tasks
            for i, t in enumerate(all_tasks):
                if t not in tasks:
                    continue
                labels[t] = batch[i+1].to(device)
                
            # Scaling the loss functions based on the algorithm choice
            loss_data_main = {}
            loss_data_base = {}
            reg_loss_data = {}
            mask = None
            optimizer_main.zero_grad()
            optimizer_base.zero_grad()
            rep_main, _ = model_main['rep'](images, mask)
            rep_base, _ = model_base['rep'](images, mask)
            for i, t in enumerate(tasks):
                # Compute gradients of each loss function wrt parameters
                out_t_main, _ = model_main[t](rep_main, None)
                out_t_base, _ = model_base[t](rep_base, None)
                loss_t_main = loss_fn[t](out_t_main, labels[t])
                loss_t_base = loss_fn[t](out_t_base, labels[t])
                loss_data_main[t] = loss_t_main.data.item()
                loss_data_base[t] = loss_t_base.data.item()
                if i > 0:
                    loss_main += loss_t_main / task_num
                    loss_base += loss_t_base / task_num
                else:
                    loss_main = loss_t_main / task_num
                    loss_base = loss_t_base / task_num
                mask_pos = labels[t] == 1
                mask_neg = labels[t] == 0
                num_pos = torch.sum(mask_pos)
                num_neg = torch.sum(mask_neg)
                if num_pos > C and num_neg > C:
                    mean_pos = torch.mean(rep_main[mask_pos], axis=0, keepdim=True)
                    mean_neg = torch.mean(rep_main[mask_neg], axis=0, keepdim=True)
                    num = num_pos + num_neg
                    #task_reg = (num_pos * torch.linalg.matrix_norm(torch.cov(torch.transpose(rep_main[mask_pos] - mean_pos, 0, 1)), ord=ord) + num_neg * torch.linalg.matrix_norm(torch.cov(torch.transpose(rep_main[mask_neg] - mean_neg, 0, 1)), ord=ord)) / num
                    task_reg = (num_pos * torch.sum(torch.square(rep_main[mask_pos] - mean_pos)) / (num_pos - 1) + num_neg * torch.sum(torch.square(rep_main[mask_neg] - mean_neg)) / (num_neg - 1)) / num
                    task_reg = task_reg / torch.square(torch.linalg.vector_norm(mean_pos - mean_neg, ord=2))
                    #task_reg = torch.reshape(task_reg, (1, 1))
                    #loss += model_main[t + "_weight"](task_reg)[0][0]
                    task_reg = rw * task_reg
                    reg_loss_data[t] = task_reg.data.item()
                    if count > 0:
                        reg += task_reg
                    else:
                        reg = task_reg
                    count += 1
            if count > 0:
                reg = reg / count
                loss_main += reg
            loss_main.backward()
            optimizer_main.step()
            loss_base.backward()
            optimizer_base.step()
            """# Scaled back-propagation
            optimizer.zero_grad()
            rep, _ = model_main['rep'](images, mask)
            for i, t in enumerate(tasks):
                out_t, _ = model_main[t](rep, masks[t])
                loss_t = loss_fn[t](out_t, labels[t])
                loss_data[t] = loss_t.data[0]
                if i > 0:
                    loss = loss + scale[t]*loss_t
                else:
                    loss = scale[t]*loss_t
            loss.backward()
            optimizer.step()"""

            writer.add_scalar('training_loss_main', loss_main.data.item(), n_iter)
            writer.add_scalar('training_reg_loss', reg.data.item(), n_iter)
            writer.add_scalar('training_loss_base', loss_base.data.item(), n_iter)
            for t in tasks:
                writer.add_scalar('training_loss_main_{}'.format(t), loss_data_main[t], n_iter)
                writer.add_scalar('training_reg_loss', reg_loss_data[t], n_iter)
                writer.add_scalar('training_loss_base_{}'.format(t), loss_data_base[t], n_iter)
            

        for m in model_main:
            model_main[m].eval()
            model_base[m].eval()

        print("-----done with training update for epoch %d-----" %(epoch + 1))
        tot_loss_main = {}
        tot_loss_base = {}
        tot_loss_main['all'] = 0.0
        tot_loss_base['all'] = 0.0
        met_main = {}
        met_base = {}
        for t in tasks:
            tot_loss_main[t] = 0.0
            met_main[t] = 0.0
            tot_loss_base[t] = 0.0
            met_base[t] = 0.0

        num_val_batches = 0
        for batch_val in val_loader:
            with torch.no_grad():
                val_images = batch_val[0].to(device)
                labels_val = {}

                for i, t in enumerate(all_tasks):
                    if t not in tasks:
                        continue
                    labels_val[t] = batch_val[i+1].to(device)

                val_rep_main, _ = model_main['rep'](val_images, None)
                val_rep_base, _ = model_base['rep'](val_images, None)
                for t in tasks:
                    out_t_main_val, _ = model_main[t](val_rep_main, None)
                    out_t_base_val, _ = model_base[t](val_rep_base, None)
                    loss_t_main = loss_fn[t](out_t_main_val, labels_val[t])
                    loss_t_base = loss_fn[t](out_t_base_val, labels_val[t])
                    mask_pos = labels_val[t] == 1
                    mask_neg = labels_val[t] == 0
                    num_pos = torch.sum(mask_pos)
                    num_neg = torch.sum(mask_neg)
                    if num_pos > C and num_neg > C:
                        mean_pos = torch.mean(val_rep_main[mask_pos], axis=0, keepdim=True)
                        mean_neg = torch.mean(val_rep_main[mask_neg], axis=0, keepdim=True)
                        num = num_pos + num_neg
                        #task_reg = (num_pos * torch.linalg.matrix_norm(torch.cov(torch.transpose(val_rep_main[mask_pos] - mean_pos, 0, 1)), ord=ord) + num_neg * torch.linalg.matrix_norm(torch.cov(torch.transpose(val_rep_main[mask_neg] - mean_neg, 0, 1)), ord=ord)) / num
                        task_reg = (num_pos * torch.sum(torch.square(val_rep_main[mask_pos] - mean_pos)) / (num_pos - 1) + num_neg * torch.sum(torch.square(val_rep_main[mask_neg] - mean_neg)) / (num_neg - 1)) / num
                        task_reg = task_reg / torch.square(torch.linalg.vector_norm(mean_pos - mean_neg, ord=2))
                        #task_reg = torch.reshape(task_reg, (1, 1))
                        #loss_t += model_main[t + "_weight"](task_reg)[0][0]
                        loss_t_main += rw * task_reg
                    tot_loss_main['all'] += loss_t_main.data.item() / task_num
                    tot_loss_main[t] += loss_t_main.data.item() / task_num
                    tot_loss_base['all'] += loss_t_base.data.item() / task_num
                    tot_loss_base[t] += loss_t_base.data.item() / task_num
                    metric_main[t].update(out_t_main_val, labels_val[t])
                    metric_base[t].update(out_t_base_val, labels_val[t])
                num_val_batches+=1
            
        mean_main_val_acc = 0.0
        mean_base_val_acc = 0.0
        for t in tasks:
            writer.add_scalar('validation_loss_main_{}'.format(t), tot_loss_main[t]/num_val_batches, n_iter)
            writer.add_scalar('validation_loss_base_{}'.format(t), tot_loss_base[t]/num_val_batches, n_iter)
            metric_main_results = metric_main[t].get_result()
            metric_base_results = metric_base[t].get_result()
            for metric_key in metric_main_results:
                writer.add_scalar('validation_metric_main_{}_{}'.format(metric_key, t), metric_main_results[metric_key], n_iter)
                writer.add_scalar('validation_metric_base_{}_{}'.format(metric_key, t), metric_base_results[metric_key], n_iter)
                mean_main_val_acc += metric_main_results[metric_key]
                mean_base_val_acc += metric_base_results[metric_key]
            metric_main[t].reset()
            metric_base[t].reset()
        writer.add_scalar('validation_loss_main', tot_loss_main['all']/len(val_dst), n_iter)
        writer.add_scalar('validation_loss_base', tot_loss_base['all']/len(val_dst), n_iter)
        mean_main_val_acc = mean_main_val_acc / task_num
        mean_base_val_acc = mean_base_val_acc / task_num
        writer.add_scalar('validation_metric_main_{}'.format(metric_key), mean_main_val_acc, n_iter)
        writer.add_scalar('validation_metric_base_{}'.format(metric_key), mean_base_val_acc, n_iter)
        print("Mean Validation set accuracy after %d epochs for the main and base models respectively is : %f and %f" %(epoch + 1, mean_main_val_acc, mean_base_val_acc))

        num_test_batches = 0
        for batch_test in test_loader:
            with torch.no_grad():
                test_images = batch_test[0].to(device)
                labels_test = {}

                for i, t in enumerate(all_tasks):
                    if t not in tasks:
                        continue
                    labels_test[t] = batch_test[i+1].to(device)

                test_rep_main, _ = model_main['rep'](test_images, None)
                test_rep_base, _ = model_base['rep'](test_images, None)
                for t in tasks:
                    out_t_main_test, _ = model_main[t](test_rep_main, None)
                    out_t_base_test, _ = model_base[t](test_rep_base, None)
                    metric_main[t].update(out_t_main_test, labels_test[t])
                    metric_base[t].update(out_t_base_test, labels_test[t])
                num_test_batches+=1
            
        mean_main_test_acc = 0.0
        mean_base_test_acc = 0.0
        for t in tasks:
            metric_main_results = metric_main[t].get_result()
            metric_base_results = metric_base[t].get_result()
            for metric_key in metric_main_results:
                writer.add_scalar('test_metric_main_{}_{}'.format(metric_key, t), metric_main_results[metric_key], n_iter)
                writer.add_scalar('test_metric_base_{}_{}'.format(metric_key, t), metric_base_results[metric_key], n_iter)
                mean_main_test_acc += metric_main_results[metric_key]
                mean_base_test_acc += metric_base_results[metric_key]
            metric_main[t].reset()
            metric_base[t].reset()
        mean_main_test_acc = mean_main_test_acc / task_num
        mean_base_test_acc = mean_base_test_acc / task_num
        writer.add_scalar('test_metric_main_{}'.format(metric_key), mean_main_test_acc, n_iter)
        writer.add_scalar('test_metric_base_{}'.format(metric_key), mean_base_test_acc, n_iter)
        print("Mean Test set accuracy after %d epochs for the main and base models respectively is : %f and %f" %(epoch + 1, mean_main_test_acc, mean_base_test_acc))

        if epoch % 3 == 0:
            # Save after every 3 epoch
            state = {'epoch': epoch+1,
                    'model_rep': model_main['rep'].state_dict(),
                    'optimizer_state' : optimizer_main.state_dict()}
            for t in tasks:
                key_name = 'model_{}'.format(t)
                state[key_name] = model_main[t].state_dict()

            torch.save(state, "saved_models/{}_{}_model.pkl".format(params['exp_id'], epoch+1))

        end = timer()
        print('Epoch ended in {}s'.format(end - start))


if __name__ == '__main__':
    train_multi_task()