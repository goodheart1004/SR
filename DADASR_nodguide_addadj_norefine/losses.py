import torch.nn.functional as F


def get_loss(output, sample, loss_name='rmse'):
    y_pred = output['y_pred']
    y, mask_hr = (sample[k] for k in ('y', 'mask_hr'))

    l1_loss = l1_loss_func(y_pred, y, mask_hr)
    mse_loss = mse_loss_func(y_pred, y, mask_hr)
    rmse_loss = rmse_loss_func(y_pred, y, mask_hr)

    if loss_name == 'l1':
        loss = l1_loss
    elif loss_name == 'rmse':
        loss = rmse_loss
    else:
        raise ValueError(f'Unsupported loss {loss_name}')

    return loss, {
        'l1_loss': l1_loss.detach(),
        'mse_loss': mse_loss.detach(),
        'rmse_loss': rmse_loss.detach(),
        'optimization_loss': loss.detach(),
    }


def mse_loss_func(pred, gt, mask):
    return F.mse_loss(pred[mask == 1.], gt[mask == 1.])


def rmse_loss_func(pred, gt, mask):
    return mse_loss_func(pred, gt, mask).sqrt()


def l1_loss_func(pred, gt, mask):
    return F.l1_loss(pred[mask == 1.], gt[mask == 1.])
