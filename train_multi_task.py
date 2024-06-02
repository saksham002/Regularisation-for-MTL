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

def get_covariance_matrix(probs):
    return torch.cov(torch.t(probs))

def get_labels_stats(dst, loader):
    with torch.no_grad():
        labels = torch.zeros((len(params['tasks']), len(dst))).to(device)
        curr = {}
        for batch in loader:
            num = 0
            for t in params['tasks']:
                i = int(t)
                if t not in curr.keys():
                    curr[t] = 0
                batch_labels = batch[i+1].to(device)
                labels[num, curr[t] : curr[t] + batch_labels.size(0)] = batch_labels
                curr[t] += batch_labels.size(0)
                num += 1

        labels_covariance = get_covariance_matrix(torch.t(labels))
        pos_mean = torch.mean(labels, dim=1)
        return pos_mean, labels_covariance

def visualise_samples(images, labels, model_base, model_main):
        from matplotlib import pyplot as plt
        
        with torch.no_grad():
            n = images.size(0)
            for i in range(1):
                ind = random.randrange(n)
                np_image = np.array(images[ind].cpu())
                plt.imshow(np_image.transpose(1, 2, 0)[:, :, ::-1], interpolation='nearest')
                plt.show()
                for t in params['tasks']:
                    i = int(t)
                    print("Task Number:", t, end = " ")
                    print("Label:", labels[t][ind], end = " ")
                    rep_base_image, _ = model_base['rep'](torch.unsqueeze(images[ind], 0), None)
                    rep_main_image, _ = model_main['rep'](torch.unsqueeze(images[ind], 0), None)
                    out_base_image_t, _ = model_base[t](rep_base_image, None)
                    out_main_image_t, _ = model_main[t](rep_main_image, None)
                    prob_base_image_t = F.sigmoid(out_base_image_t)
                    prob_main_image_t = F.sigmoid(out_main_image_t)
                    print("Probabilities:", prob_base_image_t, prob_main_image_t)
def train_multi_task(param_file):
    with open('configs.json') as config_params:
        configs = json.load(config_params)

    with open(param_file) as json_params:
        params = json.load(json_params)


    exp_identifier = []
    for (key, val) in params.items():
        if key != "optimizer" and key != "batch_size" and key != "lr" and key != "rw":
            continue
        exp_identifier+= ['{}={}'.format(key,val)]

    exp_identifier = '_'.join(exp_identifier)
    params['exp_id'] = exp_identifier
    
    device = torch.device('cuda:0')
    writer = SummaryWriter(log_dir='runs/{}_{}'.format(params['exp_id'], datetime.datetime.now().strftime("%I_%M%p on %B %d, %Y")))

    train_loader, train_dst, val_loader, val_dst, test_loader, test_dst = datasets.get_dataset(params, configs, device)
    pos_mean, labels_covariance = get_labels_stats(train_dst, train_loader)
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
    elif params["norm_type"] == "frobenius":
        ord = "fro"
    else:
        ord = None
    all_tasks = configs[params['dataset']]['all_tasks']
    print('Starting training with parameters \n \t{} \n'.format(str(params)))

    n_iter = 0
    loss_init = {}
    for epoch in tqdm(range(NUM_EPOCHS)):
        start = timer()
        print('Epoch {} Started'.format(epoch))
        if (epoch+1) % 10 == 0:
            # Every 30 epoch, half the LR
            for param_group in optimizer_base.param_groups:
                param_group['lr'] *= 0.8
            for param_group in optimizer_main.param_groups:
                param_group['lr'] *= 0.8
            print('Half the learning rate{}'.format(n_iter))

        for m in model_base:
            model_base[m].train()
            model_main[m].train()
        
        reg_train_base = 0.0
        reg_train_main = 0.0
        num_train_batches = 0.0
        
        for batch in train_loader:
            n_iter += 1
            # First member is always images
            images = batch[0].to(device)
            images.requires_grad = True

            labels = {}
            loss_data_base = {}
            loss_data_main = {}

            # Read all targets of all tasks
            for i, t in enumerate(tasks):
                i = int(t)
                labels[t] = batch[i+1].to(device)
               
            # Scaling the loss functions based on the algorithm choice
            mask = None
            optimizer_base.zero_grad()
            optimizer_main.zero_grad()
            rep_base, _ = model_base['rep'](images, mask)
            rep_main, _ = model_main['rep'](images, mask)
            n = images.size(0)
            k = task_num
            probs_base = torch.zeros((n, 0)).to(device=device)
            probs_main = torch.zeros((n, 0)).to(device=device)
            for i, t in enumerate(tasks):
                # Compute gradients of each loss function wrt parameters
                out_t_base, _ = model_base[t](rep_base, None)
                out_t_main, _ = model_main[t](rep_main, None)
                with torch.no_grad():
                    probs_base = torch.concat((probs_base, F.sigmoid(out_t_base)), dim=1)
                probs_main = torch.concat((probs_main, F.sigmoid(out_t_main)), dim=1)
                loss_t_base = loss_fn[t](out_t_base, labels[t])
                loss_t_main = loss_fn[t](out_t_main, labels[t])
                loss_data_base[t] = loss_t_base.data.item()
                loss_data_main[t] = loss_t_main.data.item()
                if i > 0:
                    loss_base += loss_t_base / task_num
                    loss_main += loss_t_main / task_num
                else:
                    loss_base = loss_t_base / task_num
                    loss_main = loss_t_main / task_num
            with torch.no_grad():
                probs_base_covariance = get_covariance_matrix(probs_base)
                reg_base = rw * torch.linalg.matrix_norm(probs_base_covariance - labels_covariance, 2)
            probs_main_covariance = get_covariance_matrix(probs_main)
            reg_main = rw * torch.linalg.matrix_norm(probs_main_covariance - labels_covariance, 2)
            loss_main += reg_main
            reg_train_base += reg_base.detach().item()
            reg_train_main += reg_main.detach().item()
            num_train_batches += 1
            loss_base.backward()
            optimizer_base.step()
            loss_main.backward()
            optimizer_main.step()

            writer.add_scalar('training_task_loss_base', loss_base.data.item(), n_iter)
            writer.add_scalar('training_reg_loss_base', reg_base.data.item(), n_iter)
            writer.add_scalar('training_task_loss_main', (loss_main - reg_main).data.item(), n_iter)
            writer.add_scalar('training_reg_loss_main', reg_main.data.item(), n_iter)
            for t in tasks:
                writer.add_scalar('training_loss_base_{}'.format(t), loss_data_base[t], n_iter)
                writer.add_scalar('training_loss_main_{}'.format(t), loss_data_main[t], n_iter)
        
        for m in model_base:
            model_base[m].eval()
            model_main[m].eval()

        print("-----done with training update for epoch %d-----" %(epoch + 1))
        tot_loss_base = {}
        tot_loss_main = {}
        tot_loss_base['all'] = 0.0
        tot_loss_main['all'] = 0.0
        reg_loss_base = {}
        reg_loss_main = {}
        reg_loss_base['all'] = 0.0
        reg_loss_main['all'] = 0.0
        met_base = {}
        met_main = {}
        for t in tasks:
            tot_loss_base[t] = 0.0
            met_base[t] = 0.0
            tot_loss_main[t] = 0.0
            met_main[t] = 0.0

        num_val_batches = 0
        for batch_val in val_loader:
            with torch.no_grad():
                val_images = batch_val[0].to(device)
                labels_val = {}
                for t in tasks:
                    i = int(t)
                    labels_val[t] = batch_val[i+1].to(device)
                
                x = random.random()
                if x < 0.015:
                    print("Images:")
                    visualise_samples(val_images, labels_val, model_base, model_main)
                val_rep_base, _ = model_base['rep'](val_images, None)
                val_rep_main, _ = model_main['rep'](val_images, None)
                n = val_images.size(0)
                k = task_num
                probs_base = torch.zeros((n, 0)).to(device=device)
                probs_main = torch.zeros((n, 0)).to(device=device)
                for t in tasks:
                    out_t_base_val, _ = model_base[t](val_rep_base, None)
                    out_t_main_val, _ = model_main[t](val_rep_main, None)
                    probs_base = torch.concat((probs_base, F.sigmoid(out_t_base_val)), dim=1)
                    probs_main = torch.concat((probs_main, F.sigmoid(out_t_main_val)), dim=1)
                    loss_t_base = loss_fn[t](out_t_base_val, labels_val[t])
                    loss_t_main = loss_fn[t](out_t_main_val, labels_val[t])
                    tot_loss_base['all'] += loss_t_base.item() / task_num
                    tot_loss_base[t] += loss_t_base.item() / task_num
                    tot_loss_main['all'] += loss_t_main.item() / task_num
                    tot_loss_main[t] += loss_t_main.item() / task_num
                    metric_base[t].update(out_t_base_val, labels_val[t])
                    metric_main[t].update(out_t_main_val, labels_val[t])
            probs_base_covariance = get_covariance_matrix(probs_base)
            reg_base = rw * torch.linalg.matrix_norm(probs_base_covariance - labels_covariance, 2)
            reg_loss_base['all'] += reg_base.item()
            probs_main_covariance = get_covariance_matrix(probs_main)
            reg_main = rw * torch.linalg.matrix_norm(probs_main_covariance - labels_covariance, 2)
            tot_loss_main['all'] += reg_main.item()
            reg_loss_main['all'] += reg_main.item()
            num_val_batches+=1
        mean_base_val_acc = 0.0
        mean_main_val_acc = 0.0
        for t in tasks:
            writer.add_scalar('validation_loss_base_{}'.format(t), tot_loss_base[t]/num_val_batches, n_iter)
            writer.add_scalar('validation_loss_main_{}'.format(t), tot_loss_main[t]/num_val_batches, n_iter)
            metric_base_results = metric_base[t].get_result()
            metric_main_results = metric_main[t].get_result()
            for metric_key in metric_base_results:
                writer.add_scalar('validation_metric_base_{}_{}'.format(metric_key, t), metric_base_results[metric_key], n_iter)
                writer.add_scalar('validation_metric_main_{}_{}'.format(metric_key, t), metric_main_results[metric_key], n_iter)
                mean_base_val_acc += metric_base_results[metric_key]
                mean_main_val_acc += metric_main_results[metric_key]
            metric_base[t].reset()
            metric_main[t].reset()
        writer.add_scalar('validation_task_loss_base', tot_loss_base['all']/num_val_batches, n_iter)
        writer.add_scalar('validation_task_loss_main', (tot_loss_main['all'] - reg_loss_main['all'])/num_val_batches, n_iter)
        writer.add_scalar('validation_reg_loss_base', reg_loss_base['all']/num_val_batches, n_iter)
        writer.add_scalar('validation_reg_loss_main', reg_loss_main['all']/num_val_batches, n_iter)
        mean_base_val_acc = mean_base_val_acc / task_num
        mean_main_val_acc = mean_main_val_acc / task_num
        best_base_val_acc = max(best_base_val_acc, mean_base_val_acc)
        best_main_val_acc = max(best_main_val_acc, mean_main_val_acc)
        writer.add_scalar('validation_metric_base_{}'.format(metric_key), mean_base_val_acc, n_iter)
        writer.add_scalar('validation_metric_main_{}'.format(metric_key), mean_main_val_acc, n_iter)
        print("Mean Validation set accuracy after %d epochs for the base and main models respectively is : %f and %f" %(epoch + 1, mean_base_val_acc, mean_main_val_acc))

        num_test_batches = 0
        for batch_test in test_loader:
            with torch.no_grad():
                test_images = batch_test[0].to(device)
                labels_test = {}

                for t in tasks:
                    i = int(t)
                    labels_test[t] = batch_test[i+1].to(device)

                test_rep_base, _ = model_base['rep'](test_images, None)
                test_rep_main, _ = model_main['rep'](test_images, None)
                for t in tasks:
                    out_t_base_test, _ = model_base[t](test_rep_base, None)
                    out_t_main_test, _ = model_main[t](test_rep_main, None)
                    metric_base[t].update(out_t_base_test, labels_test[t])
                    metric_main[t].update(out_t_main_test, labels_test[t])
                num_test_batches+=1
            
        mean_base_test_acc = 0.0
        mean_main_test_acc = 0.0
        for t in tasks:
            metric_base_results = metric_base[t].get_result()
            metric_main_results = metric_main[t].get_result()
            for metric_key in metric_base_results:
                writer.add_scalar('test_metric_base_{}_{}'.format(metric_key, t), metric_base_results[metric_key], n_iter)
                writer.add_scalar('test_metric_main_{}_{}'.format(metric_key, t), metric_main_results[metric_key], n_iter)
                mean_base_test_acc += metric_base_results[metric_key]
                mean_main_test_acc += metric_main_results[metric_key]
            metric_base[t].reset()
            metric_main[t].reset()
        mean_base_test_acc = mean_base_test_acc / task_num
        mean_main_test_acc = mean_main_test_acc / task_num
        best_base_test_acc = max(best_base_test_acc, mean_base_test_acc)
        best_main_test_acc = max(best_main_test_acc, mean_main_test_acc)
        writer.add_scalar('test_metric_base_{}'.format(metric_key), mean_base_test_acc, n_iter)
        writer.add_scalar('test_metric_main_{}'.format(metric_key), mean_main_test_acc, n_iter)
        print("Mean Test set accuracy after %d epochs for the base and main models respectively is : %f and %f" %(epoch + 1, mean_base_test_acc, mean_main_test_acc))

        if epoch % 5 == 0:
            # Save after every 5 epoch
            state = {'epoch': epoch+1,
                    'model_rep': model_main['rep'].state_dict(),
                    'optimizer_state' : optimizer_main.state_dict()}
            for t in tasks:
                key_name = 'model_{}'.format(t)
                state[key_name] = model_main[t].state_dict()

            torch.save(state, "saved_models/{}_{}_{}_model.pkl".format(tasks, rw, epoch+1))

        end = timer()
        print('Epoch ended in {}s'.format(end - start))
    print(rw, best_base_val_acc, best_main_val_acc, best_base_test_acc, best_main_test_acc sep=" ")


if __name__ == '__main__':
    train_multi_task()
