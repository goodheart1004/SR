import torch.nn.functional as F


def get_loss(output, sample, loss_name='rmse'):
    y_pred = output['y_pred']
    y, mask_hr = (sample[k] for k in ('y', 'mask_hr'))

    l1_loss, mse_loss, rmse_loss = masked_loss_triplet(y_pred, y, mask_hr)
    loss = select_loss(loss_name, l1_loss, rmse_loss)
    optimization_loss = loss
    loss_dict = {
        'l1_loss': l1_loss.detach(),
        'mse_loss': mse_loss.detach(),
        'rmse_loss': rmse_loss.detach(),
    }

    if 'y_refined' in output:
        ref_l1, ref_mse, ref_rmse = masked_loss_triplet(output['y_refined'], y, mask_hr)
        refinement_loss = select_loss(loss_name, ref_l1, ref_rmse)
        optimization_loss = optimization_loss + refinement_loss
        loss_dict.update({
            'refinement_l1_loss': ref_l1.detach(),
            'refinement_mse_loss': ref_mse.detach(),
            'refinement_rmse_loss': ref_rmse.detach(),
            'refinement_optimization_loss': refinement_loss.detach(),
        })

    loss_dict['optimization_loss'] = optimization_loss.detach()
    return optimization_loss, loss_dict


def masked_loss_triplet(pred, gt, mask):
    mse_loss = mse_loss_func(pred, gt, mask)
    return l1_loss_func(pred, gt, mask), mse_loss, mse_loss.sqrt()


def select_loss(loss_name, l1_loss, rmse_loss):
    if loss_name == 'l1':
        return l1_loss
    if loss_name == 'rmse':
        return rmse_loss
    raise ValueError(f'Unsupported loss {loss_name}')


def mse_loss_func(pred, gt, mask):
    valid = mask == 1.
    if not valid.any().item():
        return pred.sum() * 0.
    return F.mse_loss(pred[valid], gt[valid])


def rmse_loss_func(pred, gt, mask):
    return mse_loss_func(pred, gt, mask).sqrt()


def l1_loss_func(pred, gt, mask):
    valid = mask == 1.
    if not valid.any().item():
        return pred.sum() * 0.
    return F.l1_loss(pred[valid], gt[valid])
