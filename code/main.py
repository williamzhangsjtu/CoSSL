import torch
import numpy as np
import pandas as pd
import argparse 
import os
import time
import utils
import losses
import models as MODEL
from dataloader import create_dataloader

from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Average
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor

parser = argparse.ArgumentParser()
parser.add_argument('-d', "--debug", action="store_true")
parser.add_argument('-s', "--seed", default=0, type=int)
parser.add_argument('-c', "--config", type=str, default="config/config.yaml")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


def train():
    config = utils.get_config(args.config)
    out_dir = os.path.join(
        config['out_dir'], config['model'],
        time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    )
    try:
        os.makedirs(out_dir)
    except IOError:
        pass
    torch.save({"config": config}, os.path.join(out_dir, "run_config.d"))
    log_file = os.path.join(out_dir, "logging.txt")
    logger = utils.gen_logger(log_file)

    logger.info("<==== Experiments for SSL ====>")
    logger.info("output directory: {}".format(out_dir))
    for k, v in config.items():
        logger.info("{}: {}".format(k, v))

    train_ids, dev_ids = utils.get_index(
        config['ref_h5'], debug=args.debug)

    process_fn = utils.process_fn(**config['audio_args'])
    trainloader = create_dataloader(
        config['audio_h5'], config['ref_h5'], process_fn,
        index=train_ids, **config['trainloader_args']) 
    devloader = create_dataloader(
        config['audio_h5'], config['ref_h5'], process_fn,
        index=dev_ids, **config['devloader_args']) 

    model = getattr(MODEL, config['model'])(**config['model_args'])
    model = model.to(device)

    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optimizer_args'])

    scheduler = getattr(torch.optim.lr_scheduler, config['scheduler'])(
        optimizer, **config['scheduler_args'])

    criterion = getattr(losses, config['criterion'])(
        **config['criterion_args'])


    def _train(trainer, batch):
        with torch.enable_grad():
            feats, ref_feats, indices = batch
            feats, ref_feats, indices = convert_tensor(feats, device),\
                convert_tensor(ref_feats, device), convert_tensor(indices, device)
            score, mask = model(feats, ref_feats, indices)
            loss = criterion(score, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.cpu().item()
    trainer = Engine(_train)

    def _evaluate(evaluator, batch):
        with torch.no_grad():
            feats, ref_feats, indices = batch
            feats, ref_feats, indices = convert_tensor(feats, device),\
                convert_tensor(ref_feats, device), convert_tensor(indices, device)
            score, mask = model(feats, ref_feats, indices)
            loss = criterion(score, mask)
        return loss.cpu().item()
    evaluator = Engine(_evaluate)

    pbar = ProgressBar(ncols=75)
    pbar.attach(trainer, 
        output_transform=lambda x: {'loss': x})
    Average().attach(evaluator, 'Loss')
    Average().attach(trainer, 'Loss')

    @trainer.on(Events.STARTED)
    def eval_scratch(trainer):
        evaluator.run(devloader)
        eval_metric = evaluator.state.metrics['Loss']
        logger.info('MSE before training: {:<5.2f}'.format(eval_metric))
        scheduler.step(eval_metric)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(trainer):
        AvgLoss = trainer.state.metrics['Loss']
        n_epoch = trainer.state.epoch
        evaluator.run(devloader)
        eval_metric = evaluator.state.metrics['Loss']
        logger.info("<=== #{:<3} Epoch ===>".format(n_epoch))
        logger.info('Training loss: {:<5.2f}'.format(AvgLoss))
        logger.info('Evaluation MSE: {:<5.2f}'.format(eval_metric))
        scheduler.step(eval_metric)

    @trainer.on(Events.EPOCH_COMPLETED(once=config['switch_hard']))
    def switch2hard(trainer):
        model.ifhard = True

    earlystopping_handler = EarlyStopping(
        patience=config['patience'], trainer=trainer,
        score_function=lambda engine: engine.state.metrics['Loss'])

    best_checkpoint_handler = ModelCheckpoint(
        dirname=out_dir, filename_prefix='eval_best',
        score_function=lambda engine: -engine.state.metrics['Loss'],
        score_name='loss', n_saved=1, 
        global_step_transform=global_step_from_engine(trainer))

    periodic_checkpoint_handler = ModelCheckpoint(
        dirname=out_dir, filename_prefix='train_periodic',
        score_function=lambda engine: -engine.state.metrics['Loss'],
        score_name='loss', n_saved=None,
        global_step_transform=global_step_from_engine(trainer))

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, earlystopping_handler)
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, best_checkpoint_handler,
        {"model": model})
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config['save_interval']),
        periodic_checkpoint_handler, {"model": model})

    trainer.run(trainloader, config['n_epochs'])


train()
