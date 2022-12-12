import logging
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from torch.optim import Adam
from tensorize import CorefDataProcessor
import util
import argparse
import time
from os.path import join
from os import remove
from metrics import CorefEvaluator
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from model import CorefModel, IncrementalCorefModel
import conll
import gc
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class Runner:
    def __init__(self, config_name, gpu_id=0, seed=None, skip_data_loading=False, log_to_file=True):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed
        self.last_save_suffix = None
        self.old_save_suffix = None

        # Set up config
        self.config = util.initialize_config(config_name, create_dirs=log_to_file)

        # Access it here so lack of it doesn't just crash us in eval
        _ = self.config['postprocess_merge_overlapping_spans']

        # Set up logger
        if log_to_file:
            log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
            logger.addHandler(logging.FileHandler(log_path, 'a'))
            logger.info('Log file path: %s' % log_path)

        # Set up seed
        if seed:
            util.set_seed(seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        # Set up data
        if not skip_data_loading:
            self.data = CorefDataProcessor(self.config)

    def initialize_model(self, saved_suffix=None):
        if self.config['incremental']:
            model = IncrementalCorefModel(self.config, self.device)
        else:
            model = CorefModel(self.config, self.device)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix)
        return model

    def train(self, model):
        conf = self.config
        logger.info(conf)
        epochs, grad_accum = conf['num_epochs'], conf['gradient_accumulation_steps']
        patience = conf['patience'] if 'patience' in conf else epochs

        model.to(self.device)
        logger.info('Model parameters:')
        for name, param in model.named_parameters():
            logger.info('%s: %s' % (name, tuple(param.shape)))

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info('Tensorboard summary path: %s' % tb_path)

        # Set up data
        examples_train, examples_dev, examples_test = self.data.get_tensor_examples()
        stored_info = self.data.get_stored_info()
        eval_frequency = conf['eval_frequency'] if 'eval_frequency' in conf else len(examples_train)
        report_frequency = conf['report_frequency'] if 'report_frequency' in conf else len(examples_train)


        # Set up optimizer and scheduler
        total_update_steps = len(examples_train) * epochs // grad_accum
        self.total_update_steps = total_update_steps
        optimizers = self.get_optimizer(model)
        schedulers = self.get_scheduler(optimizers, total_update_steps)

        # Get model parameters for grad clipping
        bert_param, task_param, incremental_param = model.get_params()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(examples_train))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Early stopping patience: %d' % patience)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1 = 0
        start_time = time.time()
        model.zero_grad()
        pbar = tqdm(total=len(examples_train) * epochs)
        evals_without_improvements = 0
        for epo in range(epochs):
            if evals_without_improvements == patience:
                break
            random.shuffle(examples_train)  # Shuffle training set
            for doc_key, example in examples_train:
                # Forward pass
                model.train()
                example_gpu = [d.to(self.device) for d in example]
                if not conf['incremental']:
                    _, loss = model(*example_gpu)
                else:
                    min_loss_chance = conf['incremental_start_global_loss_ratio']
                    loss_delta = conf['incremental_end_global_loss_ratio'] - min_loss_chance
                    global_loss_chance = loss_delta * (len(loss_history) / total_update_steps) + min_loss_chance
                    _, loss = model(
                        *example_gpu,
                        global_loss_chance=global_loss_chance,
                        teacher_forcing=conf["incremental_teacher_forcing"],
                    )

                # Backward; accumulate gradients and clip by grad norm
                if grad_accum > 1:
                    loss /= grad_accum
                if not conf["incremental"]:
                    loss.backward()
                if conf['max_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(bert_param, conf['max_grad_norm'])
                    torch.nn.utils.clip_grad_norm_(task_param, conf['max_grad_norm'])
                    torch.nn.utils.clip_grad_norm_(incremental_param, conf['max_grad_norm'])
                if not conf['incremental']:
                    loss_during_accum.append(loss.item())
                else:
                    loss_during_accum.append(loss)

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    if conf['freeze_mention_score']:
                        # Simply clear gradients
                        model.span_emb_score_ffnn.zero_grad()
                        model.span_width_score_ffnn.zero_grad()
                    for optimizer in optimizers:
                        optimizer.step()
                    model.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    
                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % report_frequency == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / report_frequency
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, report_frequency / (end_time - start_time)))
                        start_time = end_time

                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', schedulers[0].get_last_lr()[0], len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Task', schedulers[1].get_last_lr()[-1], len(loss_history))

                    # Evaluate
                    if len(loss_history) > 0 and len(loss_history) % eval_frequency == 0:
                        f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False, conll_path=self.config['conll_eval_path'], tb_writer=tb_writer)
                        if f1 > max_f1:
                            max_f1 = f1
                            self.save_model_checkpoint(model, len(loss_history))
                            self.delete_old_checkpoint()
                            evals_without_improvements = 0
                        else:
                            evals_without_improvements += 1
                            if evals_without_improvements == patience:
                                logger.info(f'F1 evaluation score did not improve for {patience} number of evaluations: Stopping early after {epo+1} epochs')
                                break

                        logger.info('Eval max f1: %.2f' % max_f1)
                        start_time = time.time()
                pbar.update()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Wrap up
        tb_writer.close()
        return loss_history

    def evaluate(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None, out_file=None):
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction = {}

        model.eval()
        for i, (doc_key, tensor_example) in tqdm(enumerate(tensor_examples), total=len(tensor_examples)):
            gold_clusters = stored_info['gold'][doc_key]
            tensor_example = tensor_example[:7]  # Strip out gold
            example_gpu = [d.to(self.device) for d in tensor_example]
            if self.config["incremental"]:
                with torch.no_grad():
                    span_starts, span_ends, mention_to_cluster_id, predicted_clusters = model(*example_gpu)
                    model.update_evaluator(
                        span_starts,
                        span_ends,
                        predicted_clusters,
                        mention_to_cluster_id,
                        gold_clusters,
                        evaluator,
                    )
                    doc_to_prediction[doc_key] = predicted_clusters
            else:
                with torch.no_grad():
                    _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*example_gpu)
                span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
                antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
                predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator)
                doc_to_prediction[doc_key] = predicted_clusters

        p, r, f = evaluator.get_prf()
        metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        if official:
            conll_results = conll.evaluate_conll(
                self.config['conll_scorer'],
                conll_path,
                doc_to_prediction,
                stored_info['subtoken_maps'],
                out_file,
                merge_overlapping_spans=self.config['postprocess_merge_overlapping_spans'],
            )
            try:
                official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
                logger.info('Official avg F1: %.4f' % official_f1)
            except TypeError: # If any of the f1s are None due to duplicate clusters etc. (may happen early in training)
                logger.info('Unable to calculate official avg F1')

        return f * 100, metrics

    def predict(self, model, tensor_examples):
        logger.info('Predicting %d samples...' % len(tensor_examples))
        model.to(self.device)
        predicted_spans, predicted_antecedents, predicted_clusters = [], [], []

        for i, tensor_example in enumerate(tensor_examples):
            tensor_example = tensor_example[:7]
            example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*example_gpu)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            clusters, mention_to_cluster_id, antecedents = model.get_predicted_clusters(span_starts, span_ends, antecedent_idx, antecedent_scores)

            spans = [(span_start, span_end) for span_start, span_end in zip(span_starts, span_ends)]
            predicted_spans.append(spans)
            predicted_antecedents.append(antecedents)
            predicted_clusters.append(clusters)

        return predicted_clusters, predicted_spans, predicted_antecedents

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        bert_param, task_param, incremental_params = model.get_params(named=True)
        grouped_bert_param = [
            {
                'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': 0.0
            }
        ]
        optimizers = [
            AdamW(grouped_bert_param, lr=self.config['bert_learning_rate'], eps=self.config['adam_eps']),
            Adam(model.get_params()[1], lr=self.config['task_learning_rate'], eps=self.config['adam_eps'], weight_decay=0),
        ]
        if self.config['incremental']:
            optimizers.append(
                Adam(
                    model.get_params()[2],
                    lr=self.config.get('incremental_learning_rate', 0),
                    eps=self.config['adam_eps'],
                    weight_decay=0
                )
            )
        return optimizers
        # grouped_parameters = [
        #     {
        #         'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': 0.0
        #     }, {
        #         'params': [p for n, p in task_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in task_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': 0.0
        #     }
        # ]
        # optimizer = AdamW(grouped_parameters, lr=self.config['task_learning_rate'], eps=self.config['adam_eps'])
        # return optimizer

    def get_scheduler(self, optimizers, total_update_steps):
        # Only warm up bert lr
        warmup_steps = int(total_update_steps * self.config['warmup_ratio'])

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
            )

        def lr_lambda_task(current_step):
            return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

        def lr_lambda_incremental(current_step):
            return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task),
        ]
        if self.config['incremental']:
            schedulers.append(
                LambdaLR(optimizers[2], lr_lambda_incremental),
            )
        return schedulers
        # return LambdaLR(optimizer, [lr_lambda_bert, lr_lambda_bert, lr_lambda_task, lr_lambda_task])

    def save_model_checkpoint(self, model, step):
        if step < self.total_update_steps / 20:
            logger.info('Skipping model saving, we are very early in training!')
            return
        self.old_save_suffix = self.last_save_suffix
        self.last_save_suffix = f'{self.name}_{self.name_suffix}_{step}'
        path_ckpt = join(self.config['log_dir'], f'model_{self.last_save_suffix}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, model, suffix, dir=None):
        if dir is None:
            dir = self.config['log_dir']
        path_ckpt = join(dir, f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)

    def delete_old_checkpoint(self):
        if self.old_save_suffix is None or ('keep_all_saved_models' in self.config \
            and self.config['keep_all_saved_models'] == True):
            return

        path_ckpt = join(self.config['log_dir'], f'model_{self.old_save_suffix}.bin')
        remove(path_ckpt)

def build_parser():
    parser = argparse.ArgumentParser(description='Train coreference models')
    parser.add_argument('config', help='Config name to use', type=str)
    parser.add_argument('gpu', help='Which GPU to use', type=int)
    parser.add_argument('--model', help='Pre-trained model to use as basis', type=str, nargs="?", default=None)
    parser.add_argument('--mention-pre-training', help='Config to use for mention pre-trianing', type=str)
    return parser

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    runner = Runner(args.config, args.gpu)
    model = runner.initialize_model()
    if args.model:
        runner.load_model_checkpoint(model, args.model)
    if args.mention_pre_training:
        from run_mentions import run_mentions
        logger.info('Performing mention pre-training')
        mention_model, mention_runner = run_mentions(args.mention_pre_training, args.gpu)
        runner.load_model_checkpoint(model, mention_runner.last_save_suffix, dir=mention_runner.config['log_dir'])
        # We need the GPU RAM to be freed before doing things with the new model
        del mention_model, mention_runner
        gc.collect()

    runner.train(model)

    if runner.last_save_suffix is None:
        runner.last_save_suffix = f'{runner.name}_{runner.name_suffix}_unsaved'
    else:
        runner.load_model_checkpoint(model, runner.last_save_suffix)
    

    examples_train, examples_dev, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()

    path_dev_pred = join(runner.config['log_dir'], f'dev_{runner.last_save_suffix}.prediction')
    path_test_pred = join(runner.config['log_dir'], f'test_{runner.last_save_suffix}.prediction')
    runner.evaluate(model, examples_dev, stored_info, 0, official=True, conll_path=runner.config['conll_eval_path'], out_file=path_dev_pred)  # Eval dev
    #model.eval_only = True
    runner.evaluate(model, examples_test, stored_info, 0, official=True, conll_path=runner.config['conll_test_path'], out_file=path_test_pred)  # Eval test
