import yaml
from trainers import RNNClassifierTrainer, \
    CNNClassifierTrainer, \
    TCNClassifierTrainer, \
    LSTMClassifierTrainer, \
    GRUClassifierTrainer

from evaluation_metrics import plot_loss
import wandb

target = {
    'rnn': RNNClassifierTrainer,
    'lstm': LSTMClassifierTrainer,
    'gru': GRUClassifierTrainer,
    'cnn': CNNClassifierTrainer,
    'tcn': TCNClassifierTrainer
}

if __name__ == '__main__':
    with open('./config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    run_name = 'Seq classification - hw3 -'
    wandb.init(config=config, project='seq-classification', name=run_name)
    for model_name in target:
        trainer = target[model_name]
        tl, vl, cms = trainer(cfg=config)
        loss_fig = plot_loss(tl, vl, show=True)
        for cm in cms:
            wandb.log({
                'CM {} {}'.format(cms.index(cm), model_name): cm
            })
        wandb.log({
            '{} Loss'.format(model_name): loss_fig,
        })
