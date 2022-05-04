import numpy as np
import torch
import tqdm
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from neural.EarlyStopping import EarlyStopping


def val(net, eval_dataloader, criterion):
    with torch.no_grad():
        net.eval()

        val_loss = []
        for i, data in enumerate(eval_dataloader, 0):
            inputs, labels = data

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            val_loss.append(loss.item())

        return np.mean(val_loss)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(net, train_dataloader, eval_dataloader, epoch_num=150, batch_size=32):
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    optimizer = Adam(net.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    early_stopping = EarlyStopping(patience=12, verbose=False)

    for epoch in range(epoch_num):
        net.train()
        running_loss = 0.0
        lr = get_lr(optimizer)

        tq = tqdm.tqdm(total=len(train_dataloader) * batch_size)
        tq.set_description(f'epoch {epoch}, lr {lr:.6f}')

        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs).squeeze()
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels)

            tq.update(batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss = val(net, eval_dataloader, criterion)
        print("\nValidation loss: ", val_loss)
        scheduler.step(val_loss)

        early_stopping(val_loss, net)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        tq.close()

    # net.load_state_dict(torch.load('checkpoint.pt'))

    print('Finished Training')