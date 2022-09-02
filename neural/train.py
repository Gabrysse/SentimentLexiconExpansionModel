import numpy as np
import torch
import tqdm
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from neural.EarlyStopping import EarlyStopping


def val(model, eval_dataloader, criterion):
    with torch.no_grad():
        model.eval()

        val_loss = []
        for i, data in enumerate(eval_dataloader, 0):
            inputs, labels = data

            prediction = model(inputs)
            loss = criterion(prediction, labels)

            val_loss.append(loss.item())

        return np.mean(val_loss)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, train_dataloader, eval_dataloader, epoch_num=150, batch_size=32):
    torch.manual_seed(15)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    early_stopping = EarlyStopping(patience=12, verbose=False)

    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0
        lr = get_lr(optimizer)

        tq = tqdm.tqdm(total=len(train_dataloader) * batch_size)
        tq.set_description(f'epoch {epoch}, lr {lr:.6f}')

        for i, data in enumerate(train_dataloader, 0):
            optimizer.zero_grad()
            inputs, labels = data
            labels = labels.reshape(-1, 1)
            prediction = model(inputs)
            # print(prediction.size())
            # print(labels.size())
            loss = criterion(prediction, labels)

            tq.update(batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        tq.close()
        val_loss = val(model, eval_dataloader, criterion)
        print("\nValidation loss: ", val_loss)
        scheduler.step(val_loss)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # model.load_state_dict(torch.load('checkpoint.pt'))
    print('Finished Training')
