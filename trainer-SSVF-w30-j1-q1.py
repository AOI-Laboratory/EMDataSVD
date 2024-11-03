import sys, os, time, math
import argparse
from os.path import join, dirname, basename, splitext, abspath, exists
import logging
import logging.handlers
import pandas as pd
import h5py
import datetime
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchinfo import summary
from Model_SSVF_w30_j1_q1 import DUVModel

# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
BACKBONE = 'Model-SSVF-w30-j1-q1'
DATASET_TYPE = ['DGSM', 'Sobol', 'Random']
OCD_MAP = {'retch': 0, 'Rtop': 1, 'Rbot': 2, 'n': 3, 'height': 4, 'b': 5}
MODEL = '{}'.format(BACKBONE)

HDF5_EXT_RAW = 'hdf5_E_abs'
HDF5_EXT_SVD = 'hdf5_E_svd_abs'

FLOAT32_MIN = 1.1754944e-38
FLOAT32_MAX = 3.4028235e+38
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")


# ===================================================================================================================
# Hyperparameter
# ===================================================================================================================
DEST_DEV_REMOTE = join('dataset', 'Nearfield_abs')
num_epochs = 100000
batch_size = 64
learning_rate = 1.0e-5
weight_decay = 1.0e-4


def main(csv, ckpt):
    logger = logging.getLogger(__name__)
    logger.info('torch version: {}'.format(torch.version.__version__))
    logger.info('cudnn version: {}'.format(torch.backends.cudnn.version()))
    if torch.cuda.is_available():
      print('How many GPUs? {}'.format(torch.cuda.device_count()))
      device = torch.device('cuda')
    else:
      print('Only CPU')
      device = torch.device('cpu')

    ocd_df = pd.read_csv(csv)
    train_df = ocd_df[(ocd_df['type'] == DATASET_TYPE[0]) | (ocd_df['type'] == DATASET_TYPE[1]) | (ocd_df['type'] == DATASET_TYPE[2])]

    logger.info('DATASET_TYPE = {}'.format(DATASET_TYPE))
    logger.info('batch size = {}'.format(batch_size))
    logger.info('learning rate = {}'.format(learning_rate))
    logger.info('weight decay = {}'.format(weight_decay))

    root_dir = dirname(csv)

    train_dataset = DUVDataset(root_dir, train_df, type=0)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    val_df = pd.read_csv(csv.replace("DUV_20230428_All_SplitRandom", "DUV_20230428_All_Random_test"))
    val_dataset = DUVDataset(root_dir, val_df, type=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=32)

    nfreq = math.ceil((train_dataset.end_freq - train_dataset.start_freq) / train_dataset.step_freq)
    logger.info('start_freq = {}'.format(train_dataset.start_freq))
    logger.info('end_freq = {}'.format(train_dataset.end_freq))
    logger.info('step_freq = {}'.format(train_dataset.step_freq))
    logger.info('nfreq = {}'.format(nfreq))

    if ckpt:
        model = DUVModel()
        model.load_state_dict(torch.load(ckpt), strict=False)
    else:
        model = DUVModel()
        model.resnet_e.load_state_dict(torch.load('resnet18-f37072fd.pth'), strict=False)

    total_params = count_parameters(model)
    logger.info('Total trainable parameters = {}'.format(total_params))

    summary_str = str(summary(model, input_size=[(batch_size, 439, 30), (batch_size, 30, 439), (batch_size, 439, 30), (batch_size, 30, 439), (batch_size, 439, 30), (batch_size, 30, 439)], verbose=0))
    logger.info(summary_str)

    model = model.to(device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, decoupled_weight_decay=True)

    tensorboard_path = join(MODEL, timestamp, 'logs')
    os.makedirs(tensorboard_path, exist_ok=True)
    writer = SummaryWriter(tensorboard_path)
    # weight_histograms(writer, 0, model)

    time_consumption = list()
    best_regression_mape = FLOAT32_MAX
    for epoch in range(num_epochs):
        # os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')
        # time.sleep(5)
        total_train_loss = 0.0
        total_val_loss = 0.0

        retch_pe_list = np.array(list())
        Rtop_pe_list = np.array(list())
        Rbot_pe_list = np.array(list())
        n_pe_list = np.array(list())
        height_pe_list = np.array(list())
        b_pe_list = np.array(list())

        train_spent_time = 0.0
        val_spent_time = 0.0
        for ti, ((Ex_Us, Ex_V, Ey_Us, Ey_V, Ez_Us, Ez_V), targets) in enumerate(train_dataloader):
            start_time = time.time()
            Ex_Us = Ex_Us.to(device)
            Ex_V = Ex_V.to(device)
            Ey_Us = Ey_Us.to(device)
            Ey_V = Ey_V.to(device)
            Ez_Us = Ez_Us.to(device)
            Ez_V = Ez_V.to(device)
            targets = targets.to(device)
            pred_ocd = model(Ex_Us, Ex_V, Ey_Us, Ey_V, Ez_Us, Ez_V)
            loss = DUVLoss(pred_ocd, targets)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
            optimizer.step()

            total_train_loss = total_train_loss + loss

            if ti == 0:
                ocd_weights = torch.norm(model.enc_ocd.ocd_linear1.weight, 1)
                ocd_grads = torch.norm(model.enc_ocd.ocd_linear1.weight.grad, 1)

            with torch.no_grad():
                pred_ocd = pred_ocd.cpu().numpy()
                pred_ocd = train_dataset.denormalize_ocd(pred_ocd)
                ocd = targets.cpu().numpy()
                ocd = train_dataset.denormalize_ocd(ocd)
                diff = abs(pred_ocd - ocd)
                pe = (diff / ocd) * 100.0
                retch_pe_list = np.concatenate((retch_pe_list, np.array(pe[:, 0])))
                Rtop_pe_list = np.concatenate((Rtop_pe_list, np.array(pe[:, 1])))
                Rbot_pe_list = np.concatenate((Rbot_pe_list, np.array(pe[:, 2])))
                n_pe_list = np.concatenate((n_pe_list, np.array(pe[:, 3])))
                height_pe_list = np.concatenate((height_pe_list, np.array(pe[:, 4])))
                b_pe_list = np.concatenate((b_pe_list, np.array(pe[:, 5])))
   
            train_spent_time = train_spent_time + time.time() - start_time

        total_train_loss = total_train_loss / (ti + 1)

        retch_mape = np.mean(retch_pe_list)
        Rtop_mape = np.mean(Rtop_pe_list)
        Rbot_mape = np.mean(Rbot_pe_list)
        n_mape = np.mean(n_pe_list)
        height_mape = np.mean(height_pe_list)
        b_mape = np.mean(b_pe_list)

        retch_mape_std = np.std(retch_pe_list)
        Rtop_mape_std = np.std(Rtop_pe_list)
        Rbot_mape_std = np.std(Rbot_pe_list)
        n_mape_std = np.std(n_pe_list)
        height_mape_std = np.std(height_pe_list)
        b_mape_std = np.std(b_pe_list)

        logger.info('Epoch {}'.format(epoch + 1))
        logger.info('Train Loss =  {}'.format(total_train_loss))
        logger.info('Train MAPE (%) = <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}>'.format(retch_mape, Rtop_mape, Rbot_mape, n_mape, height_mape, b_mape))
        logger.info('Train STD (%) = <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}>'.format(retch_mape_std, Rtop_mape_std, Rbot_mape_std, n_mape_std, height_mape_std, b_mape_std))
        logger.info('weight: [{:.8f}]'.format(ocd_weights))
        logger.info('grad: [{:.8f}]'.format(ocd_grads))

        writer.add_scalar('Train Loss', loss.item(), epoch)
        writer.add_scalar('Train retch MAPE', retch_mape, epoch)
        writer.add_scalar('Train Rtop MAPE', Rtop_mape, epoch)
        writer.add_scalar('Train Rbot MAPE', Rbot_mape, epoch)
        writer.add_scalar('Train n MAPE', n_mape, epoch)
        writer.add_scalar('Train height MAPE', height_mape, epoch)
        writer.add_scalar('Train b MAPE', b_mape, epoch)
        writer.add_scalar('Train retch STD', retch_mape_std, epoch)
        writer.add_scalar('Train Rtop STD', Rtop_mape_std, epoch)
        writer.add_scalar('Train Rbot STD', Rbot_mape_std, epoch)
        writer.add_scalar('Train n STD', n_mape_std, epoch)
        writer.add_scalar('Train height STD', height_mape_std, epoch)
        writer.add_scalar('Train b STD', b_mape_std, epoch)

# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================

        retch_pe_list = np.array(list())
        Rtop_pe_list = np.array(list())
        Rbot_pe_list = np.array(list())
        n_pe_list = np.array(list())
        height_pe_list = np.array(list())
        b_pe_list = np.array(list())

        model.eval()
        with torch.no_grad():
            for vi, ((Ex_Us, Ex_V, Ey_Us, Ey_V, Ez_Us, Ez_V), targets) in enumerate(val_dataloader):
                start_time = time.time()
                Ex_Us = Ex_Us.to(device)
                Ex_V = Ex_V.to(device)
                Ey_Us = Ey_Us.to(device)
                Ey_V = Ey_V.to(device)
                Ez_Us = Ez_Us.to(device)
                Ez_V = Ez_V.to(device)
                targets = targets.to(device)
                pred_ocd = model(Ex_Us, Ex_V, Ey_Us, Ey_V, Ez_Us, Ez_V)
                loss = DUVLoss(pred_ocd, targets)
                total_val_loss = total_val_loss + loss

                pred_ocd = pred_ocd.cpu().numpy()
                pred_ocd = train_dataset.denormalize_ocd(pred_ocd)
                ocd = targets.cpu().numpy()
                ocd = train_dataset.denormalize_ocd(ocd)
                diff = abs(pred_ocd - ocd)
                pe = (diff / ocd) * 100.0
                retch_pe_list = np.concatenate((retch_pe_list, np.array(pe[:, 0])))
                Rtop_pe_list = np.concatenate((Rtop_pe_list, np.array(pe[:, 1])))
                Rbot_pe_list = np.concatenate((Rbot_pe_list, np.array(pe[:, 2])))
                n_pe_list = np.concatenate((n_pe_list, np.array(pe[:, 3])))
                height_pe_list = np.concatenate((height_pe_list, np.array(pe[:, 4])))
                b_pe_list = np.concatenate((b_pe_list, np.array(pe[:, 5])))
                
                val_spent_time = val_spent_time + time.time() - start_time

        total_val_loss = total_val_loss / (vi + 1)

        retch_mape = np.mean(retch_pe_list)
        Rtop_mape = np.mean(Rtop_pe_list)
        Rbot_mape = np.mean(Rbot_pe_list)
        n_mape = np.mean(n_pe_list)
        height_mape = np.mean(height_pe_list)
        b_mape = np.mean(b_pe_list)

        retch_mape_std = np.std(retch_pe_list)
        Rtop_mape_std = np.std(Rtop_pe_list)
        Rbot_mape_std = np.std(Rbot_pe_list)
        n_mape_std = np.std(n_pe_list)
        height_mape_std = np.std(height_pe_list)
        b_mape_std = np.std(b_pe_list)

        logger.info('Val Loss =  {}'.format(total_val_loss))
        logger.info('Val MAPE (%) = <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}>'.format(retch_mape, Rtop_mape, Rbot_mape, n_mape, height_mape, b_mape))
        logger.info('Val STD (%) = <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}>'.format(retch_mape_std, Rtop_mape_std, Rbot_mape_std, n_mape_std, height_mape_std, b_mape_std))

        writer.add_scalar('Val Loss', loss.item(), epoch)
        writer.add_scalar('Val retch MAPE', retch_mape, epoch)
        writer.add_scalar('Val Rtop MAPE', Rtop_mape, epoch)
        writer.add_scalar('Val Rbot MAPE', Rbot_mape, epoch)
        writer.add_scalar('Val n MAPE', n_mape, epoch)
        writer.add_scalar('Val height MAPE', height_mape, epoch)
        writer.add_scalar('Val b MAPE', b_mape, epoch)
        writer.add_scalar('Val retch STD', retch_mape_std, epoch)
        writer.add_scalar('Val Rtop STD', Rtop_mape_std, epoch)
        writer.add_scalar('Val Rbot STD', Rbot_mape_std, epoch)
        writer.add_scalar('Val n STD', n_mape_std, epoch)
        writer.add_scalar('Val height STD', height_mape_std, epoch)
        writer.add_scalar('Val b STD', b_mape_std, epoch)

        spent_time = train_spent_time + val_spent_time
        time_consumption.append(spent_time)

        val_mape = Rtop_mape + Rbot_mape + height_mape
        if best_regression_mape > val_mape:
            logger.info('*Epoch {} spent {} seconds. ({:.4f})'.format(epoch + 1, spent_time, best_regression_mape - val_mape))
            writer.flush()
            best_regression_mape = val_mape
            torch.save(model.state_dict(), join(MODEL, timestamp, 'model_ep-{}.pt'.format(epoch + 1)))
        else:
            logger.info('Epoch {} spent {} seconds'.format(epoch + 1, spent_time))

        logger.info('================================================')
        # torch.cuda.empty_cache()

        if epoch + 1 == 11:
            df = pd.DataFrame({'time_consumption': time_consumption})
            df.to_csv(join(MODEL, timestamp, 'training_time_consumption.csv'), index=True)

        logger.info('================================================')

    writer.close()


def DUVLoss(outputs, targets):
    L1Loss = nn.L1Loss()
    loss1 = L1Loss(outputs, targets)

    return loss1


class DUVDataset(Dataset):
    def __init__(self, root_dir, data, type):
        self.data = data
        self.root_dir = root_dir
        self.start_sigma = 0
        self.end_sigma = 10
        self.row = (879 - 1) // 2
        self.start_freq = 0
        self.end_freq = 1
        self.step_freq = 27
        self.nsample = math.ceil((self.end_freq - self.start_freq) / self.step_freq)
        self.ocd_min = np.array([0.02, 0.27, 0.2, 100.0, 2.7, 0.05])
        self.ocd_max = np.array([0.1, 0.33, 0.27, 600.0, 3.3, 0.1])
        self.ocd_norm_minus = self.ocd_min
        self.ocd_norm_div = self.ocd_max - self.ocd_min

        self.ocd = self.data.iloc[:, 3:].values.astype(np.float32)
        self.ocd = (self.ocd - self.ocd_norm_minus) / self.ocd_norm_div
        self.sn = self.data.iloc[:, 0].values

        if type == 0:
            self.dest_dev = join(DEST_DEV_REMOTE, 'E_SVD_Train')
        else:
            self.dest_dev = join(DEST_DEV_REMOTE, 'E_SVD_Val')

    def __len__(self):
        return len(self.data)

    def denormalize_ocd(self, ocd):
        return ocd * self.ocd_norm_div + self.ocd_norm_minus

    def __transform__(self, small_matrix):
        sigma = small_matrix[0]
        square_matrix = np.diag(sigma)
        U = small_matrix[1: self.row + 1]
        V = small_matrix[self.row + 1:]
        Us = np.dot(U, square_matrix)

        return torch.tensor(Us, dtype=torch.float), torch.tensor(np.swapaxes(V, 0, 1), dtype=torch.float)


    def __getitem__(self, index):
        sn = self.sn[index]
        ocd = self.ocd[index]

        with h5py.File(join(self.dest_dev, 'tsv2nd_{}.{}'.format(sn, HDF5_EXT_SVD)), "r") as f:
            Ex = f['Ex'][self.start_freq::self.step_freq, :, 0]
            Ey = f['Ey'][self.start_freq::self.step_freq, :, 0]
            Ez = f['Ez'][self.start_freq::self.step_freq, :, 0]

        Ex = np.swapaxes(Ex, 0, 1)
        Ey = np.swapaxes(Ey, 0, 1)
        Ez = np.swapaxes(Ez, 0, 1)

        Ex_Us, Ex_V = self.__transform__(Ex)
        Ey_Us, Ey_V = self.__transform__(Ey)
        Ez_Us, Ez_V = self.__transform__(Ez)

        ocd = torch.tensor(ocd, dtype=torch.float)

        return (Ex_Us, Ex_V, Ey_Us, Ey_V, Ez_Us, Ez_V), ocd


def weight_histograms_conv2d(writer, step, weights, layer_number):
    weights_shape = weights.shape
    num_kernels = weights_shape[0]
    for k in range(num_kernels):
        flattened_weights = weights[k].flatten()
        tag = f"layer_{layer_number}/kernel_{k}"
        writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, layer_number):
    flattened_weights = weights.flatten()
    tag = f"layer_{layer_number}"
    writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms(writer, step, model):
    print("Visualizing model weights...")
    # Iterate over all model layers
    for idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            weights = m.weight
            weight_histograms_conv2d(writer, step, weights, idx)
        elif isinstance(m, nn.Linear):
            weights = m.weight
            weight_histograms_linear(writer, step, weights, idx)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A program that performs')

    # add arguments
    parser.add_argument('-f', '--file',
                        help='The csv file contains available OCD data',
                        required=True)

    parser.add_argument('-c', '--ckpt',
                        help='The model checkpoint file',
                        required=False)

    args = parser.parse_args()

    if args.ckpt:
        ckpt = abspath(args.ckpt)
    else:
        ckpt = None

    log_path = join(MODEL, timestamp)
    os.makedirs(log_path, exist_ok=True)

    # ======= set up logging to file
    file_handler = logging.FileHandler(join(log_path, 'training_log.txt'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # ======= set up logging to console
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)

    # Get a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console)

    logger.info('csv file = {}'.format(abspath(args.file)))
    logger.info('checkpoint file = {}'.format(ckpt))

    main(abspath(args.file), ckpt)

    file_handler.flush()
