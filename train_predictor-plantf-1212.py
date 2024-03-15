import os
import csv
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from GameFormer.predictor import GameFormer
from torch.utils.data import DataLoader
from GameFormer.train_utils import *


def train_epoch(data_loader, model, optimizer, epoch):
    epoch_loss = []
    epoch_metrics = []
    model.train()

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:

            if epoch < args.warmup_epochs:
                keep_ratio = 1
            else:
                keep_ratio = args.keep_ratio

            # prepare data
            inputs = {
                'ego_agent_past': batch[0].to(args.device),
                'neighbor_agents_past': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'route_lanes': batch[4].to(args.device),
                'neighbors_future': batch[6].to(args.device)
            }

            ego_future = batch[5].to(args.device)
            # neighbors_future = batch[6].to(args.device)
            # neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

            # call the mdoel
            optimizer.zero_grad()
            # level_k_outputs, ego_plan = model(inputs)

            ego_multi_plan, probability, prediction, neighbors_future = model(inputs, keep_ratio)
            neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

            ade = torch.norm(ego_multi_plan[..., :2] - ego_future[:, None, :, :2], dim=-1)
            best_mode = torch.argmin(ade.sum(-1), dim=-1)
            best_traj = ego_multi_plan[torch.arange(ego_multi_plan.shape[0]), best_mode]
            ego_reg_loss = F.smooth_l1_loss(best_traj, ego_future)
            ego_cls_loss = F.cross_entropy(probability, best_mode.detach())
            neighbors_valid = neighbors_future_valid[:, :, :, 0]

            agent_reg_loss = F.smooth_l1_loss(
                prediction[neighbors_valid], neighbors_future[neighbors_valid][:, :2]
            )

            loss = ego_reg_loss + ego_cls_loss + agent_reg_loss

            # loss backward
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            # compute metrics
            metrics = motion_metrics(best_traj, prediction, ego_future, neighbors_future, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    # show metrics
    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    planningAHE, planningFHE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 4]), np.mean(epoch_metrics[:, 5])
    epoch_metrics = [planningADE, planningFDE, planningAHE, planningFHE, predictionADE, predictionFDE]
    logging.info(f"plannerADE: {planningADE:.4f}, plannerFDE: {planningFDE:.4f}, " +
                 f"plannerAHE: {planningAHE:.4f}, plannerFHE: {planningFHE:.4f}, " +
                 f"predictorADE: {predictionADE:.4f}, predictorFDE: {predictionFDE:.4f}\n")
        
    return np.mean(epoch_loss), epoch_metrics


def valid_epoch(data_loader, model, epoch):
    epoch_loss = []
    epoch_metrics = []
    model.eval()

    with tqdm(data_loader, desc="Validation", unit="batch") as data_epoch:
        for batch in data_epoch:

            # Token-Level Function
            if epoch < args.warmup_epochs:
                keep_ratio = 1
            else:
                keep_ratio = args.keep_ratio

           # prepare data
            inputs = {
                'ego_agent_past': batch[0].to(args.device),
                'neighbor_agents_past': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'route_lanes': batch[4].to(args.device),
                'neighbors_future': batch[6].to(args.device)
            }

            ego_future = batch[5].to(args.device)
            # neighbors_future = batch[6].to(args.device)
            # neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

            # call the mdoel
            with torch.no_grad():
                # level_k_outputs, ego_plan = model(inputs)

                ego_multi_plan, probability, prediction, neighbors_future = model(inputs, keep_ratio)
                neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

                ade = torch.norm(ego_multi_plan[..., :2] - ego_future[:, None, :, :2], dim=-1)
                best_mode = torch.argmin(ade.sum(-1), dim=-1)
                best_traj = ego_multi_plan[torch.arange(ego_multi_plan.shape[0]), best_mode]
                ego_reg_loss = F.smooth_l1_loss(best_traj, ego_future)
                ego_cls_loss = F.cross_entropy(probability, best_mode.detach())
                neighbors_valid = neighbors_future_valid[:, :, :, 0]

                agent_reg_loss = F.smooth_l1_loss(
                    prediction[neighbors_valid], neighbors_future[neighbors_valid][:, :2]
                )

                loss = ego_reg_loss + ego_cls_loss + agent_reg_loss

            # compute metrics
            metrics = motion_metrics(best_traj, prediction, ego_future, neighbors_future, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    planningAHE, planningFHE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 4]), np.mean(epoch_metrics[:, 5])
    epoch_metrics = [planningADE, planningFDE, planningAHE, planningFHE, predictionADE, predictionFDE]
    logging.info(f"val-plannerADE: {planningADE:.4f}, val-plannerFDE: {planningFDE:.4f}, " +
                 f"val-plannerAHE: {planningAHE:.4f}, val-plannerFHE: {planningFHE:.4f}, " +
                 f"val-predictorADE: {predictionADE:.4f}, val-predictorFDE: {predictionFDE:.4f}\n")

    return np.mean(epoch_loss), epoch_metrics


def model_training():
    # Logging
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(args.device))

    # set seed
    set_seed(args.seed)

    # set up model
    gameformer = GameFormer(encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels, neighbors=args.num_neighbors)
    gameformer = gameformer.to(args.device)
    logging.info("Model Params: {}".format(sum(p.numel() for p in gameformer.parameters())))

    # set up optimizer
    # optimizer = optim.AdamW(gameformer.parameters(), weight_decay = args.weight_decay, lr=args.learning_rate)
    optimizer = optim.AdamW(gameformer.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 12, 14, 16, 18], gamma=0.5)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders
    train_set = DrivingData(args.train_set + '/*.npz', args.num_neighbors)
    valid_set = DrivingData(args.valid_set + '/*.npz', args.num_neighbors)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))
    
    # begin training
    for epoch in range(train_epochs):
        logging.info(f"Epoch {epoch+1}/{train_epochs}")
        train_loss, train_metrics = train_epoch(train_loader, gameformer, optimizer, epoch)
        val_loss, val_metrics = valid_epoch(valid_loader, gameformer, epoch)

        # save to training log
        log = {'epoch': epoch+1, 'loss': train_loss, 'lr': optimizer.param_groups[0]['lr'], 'val-loss': val_loss, 
               'train-planningADE': train_metrics[0], 'train-planningFDE': train_metrics[1], 
               'train-planningAHE': train_metrics[2], 'train-planningFHE': train_metrics[3], 
               'train-predictionADE': train_metrics[4], 'train-predictionFDE': train_metrics[5],
               'val-planningADE': val_metrics[0], 'val-planningFDE': val_metrics[1], 
               'val-planningAHE': val_metrics[2], 'val-planningFHE': val_metrics[3],
               'val-predictionADE': val_metrics[4], 'val-predictionFDE': val_metrics[5]}

        if epoch == 0:
            with open(f'./training_log/{args.name}/train_log.csv', 'w') as csv_file: 
                writer = csv.writer(csv_file) 
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(f'./training_log/{args.name}/train_log.csv', 'a') as csv_file: 
                writer = csv.writer(csv_file)
                writer.writerow(log.values())

        # reduce learning rate
        scheduler.step()

        # save model at the end of epoch
        torch.save(gameformer.state_dict(), f'training_log/{args.name}/model_epoch_{epoch+1}_valADE_{val_metrics[0]:.4f}.pth')
        logging.info(f"Model saved in training_log/{args.name}\n")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument('--train_set', type=str, help='path to train data')
    parser.add_argument('--valid_set', type=str, help='path to validation data')
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--encoder_layers', type=int, help='number of encoding layers', default=3)
    parser.add_argument('--decoder_levels', type=int, help='levels of reasoning', default=1)
    parser.add_argument('--num_neighbors', type=int, help='number of neighbor agents to predict', default=20)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=40)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=32)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 1e-4)', default=1e-4)
    parser.add_argument('--weight_decay', type=float, help='weight decay (default: 1e-2)', default=1e-2)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    parser.add_argument('--warmup_epochs', type=int, help='warmup epochs', default=20)
    parser.add_argument('--keep_ratio', type=float, help='keep ratio', default=0.9)
    args = parser.parse_args()
    # Run
    model_training()