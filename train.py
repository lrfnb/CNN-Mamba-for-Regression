from __future__ import print_function
import os
import numpy as np
from math import sqrt
from math import log10
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from utils.timer import Timer
from utils.logger import Logger
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class WeightedResolutionLoss:
    def __init__(self):
        self.high_res_weight = 18.0
        self.medium_res_weight = 3.0
        self.low_res_weight = 0.8

        self.last_loss_stats = {}

    def compute_weighted_loss(self, predictions, labels):
        batch_size = predictions.size(0)

        high_res_pred = predictions[:, 0]
        medium_res_pred = predictions[:, 1]
        low_res_pred = predictions[:, 2]

        high_res_label = labels[:, 0]
        medium_res_label = labels[:, 1]
        low_res_label = labels[:, 2]

        high_loss_raw = F.l1_loss(high_res_pred, high_res_label, reduction='mean')
        medium_loss_raw = F.l1_loss(medium_res_pred, medium_res_label, reduction='mean')
        low_loss_raw = F.l1_loss(low_res_pred, low_res_label, reduction='mean')

        high_loss_weighted = high_loss_raw * self.high_res_weight
        medium_loss_weighted = medium_loss_raw * self.medium_res_weight
        low_loss_weighted = low_loss_raw * self.low_res_weight

        total_loss = high_loss_weighted + medium_loss_weighted + low_loss_weighted
        raw_total = high_loss_raw + medium_loss_raw + low_loss_raw

        self.last_loss_stats = {
            'total_loss': total_loss.item(),
            'high_loss_raw': high_loss_raw.item(),
            'medium_loss_raw': medium_loss_raw.item(),
            'low_loss_raw': low_loss_raw.item(),
            'high_loss_weighted': high_loss_weighted.item(),
            'medium_loss_weighted': medium_loss_weighted.item(),
            'low_loss_weighted': low_loss_weighted.item(),
            'high_ratio_raw': high_loss_raw.item() / raw_total.item() * 100,
            'medium_ratio_raw': medium_loss_raw.item() / raw_total.item() * 100,
            'low_ratio_raw': low_loss_raw.item() / raw_total.item() * 100,
            'high_ratio_weighted': high_loss_weighted.item() / total_loss.item() * 100,
            'medium_ratio_weighted': medium_loss_weighted.item() / total_loss.item() * 100,
            'low_ratio_weighted': low_loss_weighted.item() / total_loss.item() * 100
        }

        return total_loss

    def get_last_stats(self):
        return self.last_loss_stats


weighted_loss_fn = WeightedResolutionLoss()

if __name__ == '__main__':

    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


    def load_img(filepath):
        image = Image.open(filepath).convert('RGB')
        return image


    def rgb2y_matlab(x):
        K = np.array([65.481, 128.553, 24.966]) / 255.0
        Y = 16 + np.matmul(x, K)
        return Y.astype(np.uint8)


    opt = TrainOptions().parse()

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    print("model is ", model)

    logger = Logger(opt)
    timer = Timer()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    best_loss = float('inf')
    loss_stats_history = []

    single_epoch_iters = (dataset_size // opt.batch_size)
    total_iters = opt.total_epochs * single_epoch_iters
    cur_iters = opt.resume_iter + opt.resume_epoch * single_epoch_iters

    print('=' * 80)
    print(f'Training Configuration:')
    print(f'  Total Epochs: {opt.total_epochs}')
    print(f'  Resume from Epoch: {opt.resume_epoch}, Iter: {opt.resume_iter}')
    print(f'  Iterations per Epoch: {single_epoch_iters}')
    print(f'  Total Iterations: {total_iters}')
    print(f'  Starting from Iteration: {cur_iters}')
    print('=' * 80)

    try:
        for epoch in range(opt.resume_epoch, opt.total_epochs + 1):
            print('\n' + '=' * 80)
            print(f'Starting Epoch {epoch}/{opt.total_epochs}')
            print('=' * 80)

            epoch_loss_stats = []

            iter_start = opt.resume_iter if epoch == opt.resume_epoch else 0

            for i, data in enumerate(dataset, start=iter_start):
                cur_iters += 1
                logger.set_current_iter(cur_iters)
                model.set_input(data, cur_iters)
                timer.update_time('DataTime')

                model.forward()
                timer.update_time('Forward')

                if hasattr(model, 'predicted_resolutions') and hasattr(model, 'target_resolutions'):
                    custom_loss = weighted_loss_fn.compute_weighted_loss(
                        model.predicted_resolutions,
                        model.target_resolutions
                    )

                    model.optimizer_R.zero_grad()
                    custom_loss.backward()
                    model.optimizer_R.step()

                    loss_stats = weighted_loss_fn.get_last_stats()
                    epoch_loss_stats.append(loss_stats)

                    model.loss_Total = custom_loss
                    model.loss_MSE = torch.tensor(loss_stats['high_loss_weighted'], device=custom_loss.device)
                    model.loss_MAE = torch.tensor(loss_stats['medium_loss_weighted'], device=custom_loss.device)
                    model.loss_SmoothL1 = torch.tensor(loss_stats['low_loss_weighted'], device=custom_loss.device)

                else:
                    model.optimize_parameters()

                timer.update_time('Backward')

                loss = model.get_current_losses()

                if hasattr(model, 'predicted_resolutions') and hasattr(model, 'target_resolutions'):
                    loss_stats = weighted_loss_fn.get_last_stats()
                    loss['High'] = loss_stats['high_loss_weighted']
                    loss['Medium'] = loss_stats['medium_loss_weighted']
                    loss['Low'] = loss_stats['low_loss_weighted']

                loss.update(model.get_lr())
                logger.record_losses(loss)

                current_loss = loss.get('Total', None) or loss.get('loss_Total', None) or loss.get('loss', None) or \
                               list(loss.values())[0]

                if isinstance(current_loss, torch.Tensor):
                    current_loss = current_loss.item()

                if current_loss < best_loss:
                    best_loss = current_loss
                    print(f'\n New best loss: {best_loss:.6f} at epoch {epoch}, iter {i}')

                    info = {'resume_epoch': epoch, 'resume_iter': i + 1, 'best_loss': best_loss}
                    model.save_networks('best_loss', info)

                if cur_iters % opt.print_freq == 0:
                    print('\n' + '-' * 80)
                    print(f'Model log directory: {opt.expr_dir}')
                    print(f'Current loss: {current_loss:.6f}, Best loss: {best_loss:.6f}')

                    if epoch_loss_stats:
                        avg_stats = {}
                        for key in epoch_loss_stats[0].keys():
                            avg_stats[key] = np.mean([stats[key] for stats in epoch_loss_stats])

                        print(f'Average Epoch Stats:')
                        print(
                            f'  High Res   (×{weighted_loss_fn.high_res_weight}): {avg_stats["high_loss_weighted"]:.6f} ({avg_stats["high_ratio_weighted"]:.1f}%)')
                        print(
                            f'  Medium Res (×{weighted_loss_fn.medium_res_weight}): {avg_stats["medium_loss_weighted"]:.6f} ({avg_stats["medium_ratio_weighted"]:.1f}%)')
                        print(
                            f'  Low Res    (×{weighted_loss_fn.low_res_weight}): {avg_stats["low_loss_weighted"]:.6f} ({avg_stats["low_ratio_weighted"]:.1f}%)')

                    epoch_progress = '{:03d}|{:05d}/{:05d}'.format(epoch, i, single_epoch_iters)
                    logger.printIterSummary(epoch_progress, cur_iters, total_iters, timer)
                    print('-' * 80)

                info = {'resume_epoch': epoch, 'resume_iter': i + 1}

                if cur_iters % 1000 == 0:
                    save_suffix = 'iter_PMamba_%d' % cur_iters
                    model.save_networks(save_suffix, info)
                    print(f' Saved checkpoint: {save_suffix}')

                if cur_iters % opt.save_latest_freq == 0:
                    print(f' Saving latest model (epoch {epoch}, iters {cur_iters})')
                    model.save_networks('latest', info)

                if opt.debug:
                    print('  Debug mode: Breaking after first batch')
                    break

            if epoch_loss_stats:
                loss_stats_history.append(epoch_loss_stats)

                avg_stats = {}
                for key in epoch_loss_stats[0].keys():
                    avg_stats[key] = np.mean([stats[key] for stats in epoch_loss_stats])

                print('\n' + '=' * 80)
                print(f'Epoch {epoch} Summary:')
                print(f'  Average Total Loss: {avg_stats["total_loss"]:.6f}')
                print(f'  Average High Loss: {avg_stats["high_loss_weighted"]:.6f}')
                print(f'  Average Medium Loss: {avg_stats["medium_loss_weighted"]:.6f}')
                print(f'  Average Low Loss: {avg_stats["low_loss_weighted"]:.6f}')
                print('=' * 80)

            if opt.debug and epoch > 5:
                print('  Debug mode: Exiting after epoch 5')
                break

            for scheduler in model.schedulers:
                scheduler.step()

            print(f' Epoch {epoch} completed!')

        print('\n' + '=' * 80)
        print(' TRAINING COMPLETED SUCCESSFULLY! ')
        print('=' * 80)
        print(f'Total Epochs Completed: {opt.total_epochs}')
        print(f'Best Loss Achieved: {best_loss:.6f}')
        print(f'Total Iterations: {cur_iters}')
        print('=' * 80)

    except KeyboardInterrupt:
        print('\n' + '=' * 80)
        print('Training interrupted by user')
        print(f'Last completed epoch: {epoch}')
        print(f'Last iteration: {cur_iters}')
        print('=' * 80)

        info = {'resume_epoch': epoch, 'resume_iter': i + 1}
        model.save_networks('interrupted', info)
        print(' Saved interrupted checkpoint')

    except Exception as e:
        print('\n' + '=' * 80)
        print(' Training failed with error:')
        print(str(e))
        print('=' * 80)
        raise

    finally:
        if loss_stats_history:
            stats_save_path = os.path.join(opt.expr_dir, 'loss_stats_history.npy')
            np.save(stats_save_path, loss_stats_history)
            print(f' Loss statistics saved to: {stats_save_path}')