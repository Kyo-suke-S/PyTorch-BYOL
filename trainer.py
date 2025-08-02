import os
import logging
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import _create_model_training_folder

logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)


class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.no_tqdm = params['no_tqdm']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml"])#, "main.py", 'trainer.py'])
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()
        logging.info(f"Start BYOL training for {self.max_epochs} epochs.")
        logging.info(f"Training with gpu: {self.device}.")

        self.online_network.train()
        self.predictor.train()

        for epoch_counter in tqdm(range(self.max_epochs), disable=self.no_tqdm):
            total_loss = 0.0
            counter = 0

            for (batch_view_1, batch_view_2), _ in train_loader:
                counter += 1

                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)

                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
                    self.writer.add_image('views_1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
                    self.writer.add_image('views_2', grid, global_step=niter)

                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1
                total_loss += loss.item()

            total_loss /= counter
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {total_loss}")
            if self.no_tqdm:
                print("End of epoch {}, loss {}".format(epoch_counter, total_loss))
        
        logging.info("Training has finished.")

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)

'''
BYOLにアテンション機構を導入アテンション
networkにprojectionが内包されていないので注意
'''
class BYOLTrainer_withAttention:
    def __init__(self, online_network, target_network, projetion, target_projetion, attention, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.projetion = projetion
        self.target_projetion = target_projetion
        self.attention = attention
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.no_tqdm = params['no_tqdm']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        self.atten_lambda = params['atten_lambda']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml"])#, "main.py", 'trainer.py'])
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.projetion.parameters(), self.target_projetion.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.projetion.parameters(), self.target_projetion.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()
        logging.info(f"Start BYOL with Attention (lambda: {self.atten_lambda}) training for {self.max_epochs} epochs.")
        logging.info(f"Training with gpu: {self.device}.")

        self.online_network.train()
        self.projetion.train()
        self.predictor.train()

        for epoch_counter in tqdm(range(self.max_epochs), disable=self.no_tqdm):
            total_loss = 0.0
            total_atten_loss = 0.0
            counter = 0

            for (batch_view_1, batch_view_2), _ in train_loader:
                counter += 1

                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)

                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
                    self.writer.add_image('views_1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
                    self.writer.add_image('views_2', grid, global_step=niter)

                loss, attn_loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1
                total_loss += loss.item()
                total_atten_loss += attn_loss.item()

            total_loss /= counter
            total_atten_loss /= counter
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {total_loss}\tAttention loss: {total_atten_loss}")
            if self.no_tqdm:
                print("End of epoch {}, loss {}, attention loss {}".format(epoch_counter, total_loss, total_atten_loss))
        
        logging.info("Training has finished.")

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        h1, ext_H1 = self.online_network(batch_view_1)
        h2, ext_H2 = self.online_network(batch_view_2)
        predictions_from_view_1 = self.predictor(self.projetion(h1))
        predictions_from_view_2 = self.predictor(self.projetion(h2))

        # compute key features
        with torch.no_grad():
            tgt_to_h2, to_ext_H2 = self.target_network(batch_view_1)
            tgt_to_h1, to_ext_H1 = self.target_network(batch_view_2)
            targets_to_view_2 = self.target_projetion(tgt_to_h2)
            targets_to_view_1 = self.target_projetion(tgt_to_h1)

        atten_targets_to_view_1, _ = self.attention(to_ext_H1, ext_H1)
        atten_targets_to_view_2, _ = self.attention(to_ext_H2, ext_H2)

        #predictions_from_view_1 = torch.cat([predictions_from_view_1, ext_H1.reshape(-1, ext_H1.size(-1))], dim=0)
        ext_H1 = ext_H1.permute(0, 2, 3, 1).flatten(1, 2)
        ext_H2 = ext_H2.permute(0, 2, 3, 1).flatten(1, 2)
        ext_predictions_from_view_1 = self.predictor(self.projetion(ext_H1.reshape(-1, ext_H1.size(-1))))

        #targets_to_view_1 = torch.cat([targets_to_view_1, atten_targets1.reshape(-1, atten_targets1.size(-1))], dim=0)
        ext_targets_to_view_1 = self.target_projetion(atten_targets_to_view_1.reshape(-1, atten_targets_to_view_1.size(-1)))

        #predictions_from_view_2 = torch.cat([predictions_from_view_2, ext_H2.reshape(-1, ext_H2.size(-1))], dim=0)
        ext_predictions_from_view_2 = self.predictor(self.projetion(ext_H2.reshape(-1, ext_H2.size(-1))))
        #targets_to_view_2 = torch.cat([targets_to_view_2, atten_targets2.reshape(-1, atten_targets2.size(-1))], dim=0)
        ext_targets_to_view_2 = self.target_projetion(atten_targets_to_view_2.reshape(-1, atten_targets_to_view_2.size(-1)))


        loss1 = self.regression_loss(predictions_from_view_1, targets_to_view_1) 
        loss2 = self.regression_loss(ext_predictions_from_view_1, ext_targets_to_view_1)
        loss1 += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        loss2 += self.regression_loss(ext_predictions_from_view_2, ext_targets_to_view_2)
        loss = loss1.mean() + self.atten_lambda*loss2.mean()
        return loss, loss2.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'attention_state_dict': self.attention.state_dict(),
            'projetion_state_dict': self.projetion.state_dict(),
            'target_projetion': self.target_projetion.state_dict(),
        }, PATH)