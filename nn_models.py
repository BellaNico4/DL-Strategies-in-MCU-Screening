import torch
from torch import nn
import pytorch_lightning as pl
import sklearn
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler
from torch.utils.data import TensorDataset,DataLoader
from torch.optim.lr_scheduler import  ReduceLROnPlateau
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from sklearn.base import BaseEstimator, TransformerMixin



class SoftOrdering1DCNN(pl.LightningModule):

    def __init__(self, input_dim, sign_size=32, cha_input=16, cha_hidden=32,
                 K=2, dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
        super().__init__()

        hidden_size = sign_size*cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size//2
        output_size = (sign_size//4) * cha_hidden

        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output

        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = nn.Conv1d(
            cha_input,
            cha_input*K,
            kernel_size=5,
            stride = 1,
            padding=2,
            groups=cha_input,
            bias=False)

        self.conv1 = nn.utils.weight_norm(conv1, dim=None)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input*K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input*K,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)
        nn.init.xavier_uniform_(self.conv2.weight)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)
        nn.init.xavier_uniform_(self.conv3.weight)

        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=cha_hidden,
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.loss = nn.MSELoss()

    def forward(self, x):

        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x =  x + x_s
        x = nn.functional.relu(x)

        x = self.avg_po_c4(x)

        x = self.flt(x)

        return x
    
    def transform(self, x):
        x = self(x)
        return x


    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_logit = self.forward(X)
        loss = self.loss(y_logit, y)
        metric = mean_absolute_error(y.cpu().numpy(), y_logit.detach().cpu().numpy())
        self.log('test_loss', loss)
        self.log('test_metric', metric)

    def configure_optimizers(self,lr=1e-4,momentum=0.7,weight_decay=0):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                min_lr=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True,
            'monitor': 'valid_loss',
        }
        return [optimizer], [scheduler]

class RegressionNN(pl.LightningModule):

    def __init__(self,feature_extractor=SoftOrdering1DCNN(input_dim=27),output_dim=1,dropout_output=0.2):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.batch_norm2 = nn.BatchNorm1d(feature_extractor.output_size)
        self.dropout2 = nn.Dropout(dropout_output)
        self.dropout_output = dropout_output
        self.dense2 = nn.Linear(feature_extractor.output_size, output_dim, bias=True)
        self.output_dim = output_dim
        self.dense2 = nn.utils.weight_norm(self.dense2)
        nn.init.xavier_uniform_(self.dense2.weight)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x


    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def set_eval_mode(self):
        self.train(False)

    def transform(self,x):
        return self.feature_extractor(x)
    
    def get_transformed_output_shape(self):
        x = torch.rand((2,27))
        return self.transform(x).shape[1]

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_logit = self.forward(X)
        loss = self.loss(y_logit, y)
        metric = mean_absolute_error(y.cpu().numpy(), y_logit.detach().cpu().numpy())
        self.log('test_loss', loss)
        self.log('test_metric', metric)

    def configure_optimizers(self,lr=1e-3,momentum=0.6,weight_decay=0):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                min_lr=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True,
            'monitor': 'valid_loss',
        }
        return [optimizer], [scheduler]


class SoftOrdering1DCNN_AutoEncoder(pl.LightningModule):

    def __init__(self, input_dim, sign_size=32, cha_input=16, cha_hidden=32,
                 K=2, dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
        super().__init__()

        hidden_size = sign_size*cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size//2
        output_size = (sign_size//4) * cha_hidden

        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output

        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = nn.Conv1d(
            cha_input,
            cha_input*K,
            kernel_size=5,
            stride = 1,
            padding=2,
            groups=cha_input,
            bias=False)

        self.conv1 = nn.utils.weight_norm(conv1, dim=None)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input*K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input*K,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)
        nn.init.xavier_uniform_(self.conv2.weight)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)
        nn.init.xavier_uniform_(self.conv3.weight)

        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=cha_hidden,
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()
        self.loss = nn.MSELoss()
        
        # 1st De-conv
        
        
        self.batch_norm_c5 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c5 = nn.Dropout(dropout_hidden)
        self.t_conv1 = nn.ConvTranspose1d(
            cha_hidden,
            cha_hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=cha_hidden,
            bias=False)
        nn.init.xavier_uniform_(self.t_conv1.weight)
        
        
        # 2nd De-conv
        self.batch_norm_c6 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c6 = nn.Dropout(dropout_hidden)
        self.t_conv2 = nn.ConvTranspose1d(
            cha_hidden,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        nn.init.xavier_uniform_(self.t_conv2.weight)
        # 3th De-conv
        self.batch_norm_c7 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c7 = nn.Dropout(dropout_hidden)
        self.t_conv3 = nn.ConvTranspose1d(
            cha_hidden,
            cha_input*K,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        nn.init.xavier_uniform_(self.t_conv3.weight)
        
        # 4st De-conv layer
        self.batch_norm_c8 = nn.BatchNorm1d(cha_input*K)
        self.t_conv4 = nn.ConvTranspose1d(
            cha_input*K,
            cha_input,
            kernel_size=5,
            stride = 1,
            padding=2,
            groups=cha_input,
            bias=False)

        nn.init.xavier_uniform_(self.t_conv4.weight)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_input)
        dense2 = nn.Linear(hidden_size, input_dim, bias=False)
        self.dense2 = nn.utils.weight_norm(dense2)
        
    def forward(self, x):
        
        ##ENCODER
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))
        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)
        
        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))
        
        x = self.ave_po_c1(x)
        
        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        
        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))
        
        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        
        x = self.avg_po_c4(x)
        
        x = self.flt(x)
        
        
        ##DECODER 
        
        x = x.reshape(x.shape[0], self.cha_hidden, self.sign_size1 // 4)
        
        x = nn.functional.interpolate(x,scale_factor=2)
        
        x = self.batch_norm_c5(x)
        x = nn.functional.relu(self.t_conv1(x))
        
        
        x = self.batch_norm_c6(x)
        x = self.dropout_c6(x)
        x = nn.functional.relu(self.t_conv2(x))
       
        
        #x =  x + x_s
        x = self.batch_norm_c7(x)
        x = self.dropout_c7(x)
        x = nn.functional.relu(self.t_conv3(x))
        
        
        x = nn.functional.interpolate(x,scale_factor=2)
        x = self.batch_norm_c8(x)
        x = self.t_conv4(x)
        
        x = nn.functional.relu(x)
        
        
        x = self.flt(x)
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = nn.functional.celu(self.dense2(x))
        
        return x

    def transform(self,x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))
        
        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)
        
        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))
        
        x = self.ave_po_c1(x)
        
        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        
        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))
        
        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        
        x = self.avg_po_c4(x)
        
        x = self.flt(x)
        return x
    
    def get_transformed_output_shape(self):
        x = torch.rand((2,27))
        return self.transform(x).shape[1]
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_logit = self.forward(X)
        #y_probs = torch.sigmoid(y_logit).detach().cpu().numpy()
        loss = self.loss(y_logit, y)
        #metric = roc_auc_score(y.cpu().numpy(), y_probs)
        metric = mean_absolute_error(y.cpu().numpy(), y_logit.detach().cpu().numpy())
        self.log('test_loss', loss)
        self.log('test_metric', metric)

    def configure_optimizers(self,lr=1e-4,momentum=0.6,weight_decay=0):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                min_lr=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True,
            'monitor': 'valid_loss',
        }
        return [optimizer], [scheduler]
     
class SoftOrderingTrasnformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer_class,parameter_dict,allow_training=False,val_ratio=0.15,random_state=42,batch_size=16,callbacks = 
                 [EarlyStopping(monitor='valid_loss',min_delta=.01,patience=30,verbose=True,mode='min')],
                 epochs=100,pre_trained_ckpt=None,device='cpu'):
        # save the features list internally in the class
        self.transformer_class = transformer_class
        self.parameter_dict = parameter_dict
        self.transformer = self.transformer_class(**parameter_dict)
        self.transformer.to(device)
        self.device = device
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.epochs = epochs
        
        self.allow_training = allow_training        
        self.pre_trained_ckpt = pre_trained_ckpt
        if self.pre_trained_ckpt is not None:
            self.transformer.load_state_dict(torch.load(self.pre_trained_ckpt),strict=True)
        self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False
        
    def fit(self, X, y = None):
        if self.allow_training:
            for param in self.transformer.parameters():
                param.requires_grad = True
            self.transformer.train()
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_ratio, random_state=self.random_state)
            
            if len(X_train) % self.batch_size == 1:
                X_train = X_train[0:-1]
                y_train = y_train[0:-1]
            if len(X_val) % self.batch_size == 1:
                X_val = X_val[0:-1]
                y_val = y_val[0:-1]
            
            train_tensor_dset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float),
                torch.tensor(y_train, dtype=torch.float)
                )

            valid_tensor_dset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float),
                torch.tensor(y_val, dtype=torch.float)
                )
            gpus = 0 if self.device == 'cpu' else 1
            trainer = pl.Trainer(callbacks=self.callbacks, min_epochs=1, max_epochs=self.epochs, gpus=gpus,enable_progress_bar = False)
           
            total_net = RegressionNN(self.transformer,output_dim=1,dropout_output=0.1)
            total_net.train()
            total_net.configure_optimizers(weight_decay=0.2,lr=1e-4)
            trainer.fit(
            total_net,
                DataLoader(train_tensor_dset, batch_size=self.batch_size, shuffle=True, num_workers=8,drop_last=False),
                DataLoader(valid_tensor_dset, batch_size=self.batch_size, shuffle=False, num_workers=8,drop_last=False),
                )
            
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False
            del total_net
            del train_tensor_dset
            del valid_tensor_dset
        return self
    
    def transform(self, X, y = None):
        # return the trasnformed X
        X_transformed = torch.tensor(X, dtype=torch.float).to(self.transformer.device)
        X_transformed = self.transformer(X_transformed).detach().cpu().numpy()
        return X_transformed
    
    
class DummySelector(BaseEstimator, TransformerMixin):
    def __init__(self,columns,random_state=None,features_len=27):
        # save the features list internally in the class
        self.random_state = random_state
        self.features_len = features_len
        self.rng =  np.random.default_rng(seed=random_state) 
        self.columns = columns
        self.features = self.rng.choice(self.columns, size=self.features_len, replace=False)
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return X[:,self.features]
    
class BestSelector(BaseEstimator, TransformerMixin):
    def __init__(self,best_columns):
        # save the features list internally in the class
        self.features_len = len(best_columns)
        self.best_columns = best_columns
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return X[:,self.best_columns]