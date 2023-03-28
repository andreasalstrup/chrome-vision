import logging
import os
import shutil
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

torch.manual_seed(0)


class CLRTrainer(object):

    def __init__(self, epochs, model, optimizer, scheduler, criterion, batch_size, device, temp, con_views, **kwargs):
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = device
        self.temp = temp
        self.con_views = con_views

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.con_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temp
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=False)

        # save config file
        #save_config_file(self.writer.log_dir, self)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.epochs} epochs.")
        logging.info(f"Training with CPU.")

        for epoch_counter in range(self.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.device)

                features = self.model(images)
                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % 0.07 == 0:
                    top1, top5 = self.accuracy(logits, labels, (1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.epochs)
        self.save_checkpoint({
            'epoch': self.epochs,
            'arch': 'ResNet18',
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        
    
    ## UTILS ##
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')


    def save_config_file(self, model_checkpoints_folder, args):
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)
            with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
                yaml.dump(args, outfile, default_flow_style=False)


    def accuracy(self, output, target, topk):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
