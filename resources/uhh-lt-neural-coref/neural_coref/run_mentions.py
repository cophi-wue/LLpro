import logging
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics
import time
from neural_coref.run import Runner
from os.path import join
from os import remove
from torch.optim.lr_scheduler import LambdaLR
from neural_coref.model import MentionModel
import conll
import sys
import csv
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class MentionRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_save_suffix = None
        self.old_save_suffix = None

    def initialize_model(self, saved_suffix=None):
        model = MentionModel(self.config, self.device)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix)
        return model

    def train(self, model):
        conf = self.config
        logger.info(conf)
        epochs, grad_accum = conf['num_epochs_mention'], conf['gradient_accumulation_steps']

        model.to(self.device)

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
        optimizers = self.get_optimizer(model)
        schedulers = self.get_scheduler(optimizers, total_update_steps)

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(examples_train))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        optimizers = self.get_optimizer(model)
        schedulers = self.get_scheduler(optimizers, total_update_steps)

        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1 = 0
        start_time = time.time()
        model.zero_grad()
        pbar = tqdm(total=len(examples_train) * epochs)
        model.train()
        for epo in range(epochs):
            random.shuffle(examples_train)  # Shuffle training set
            for doc_key, example in examples_train:
                example_gpu = [d.to(self.device) for d in example]
                loss = model(*example_gpu)[3]
                loss.backward()
                loss_history.append(loss)
                loss_during_report += loss
                for optimizer in optimizers:
                    optimizer.step()
                model.zero_grad()
                for scheduler in schedulers:
                    scheduler.step()
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
                    logger.info('Eval max f1: %.2f' % max_f1)
                    start_time = time.time()
                pbar.update()
        # Wrap up
        tb_writer.close()
        return loss_history

    def evaluate(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None, out_file=None):
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)

        total = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        scores_list = []
        label_list = []

        doc_to_prediction = {}

        negative_examples = 0.0
        positive_examples = 0.0
        recall_values_acc = torch.zeros(101, device=self.device)
        recall_actual_acc = 0

        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                scores, labels, cluster_to_spans, loss = model(*example_gpu)
                doc_to_prediction[doc_key] = cluster_to_spans.values()
                total += len(labels)
                predicted = (scores > 0.5).T.squeeze()
                labels = labels.to(torch.bool)
                num_gold_mentions = labels.sum().item()
                positive_examples += num_gold_mentions
                negative_examples += len(labels) - num_gold_mentions
                true_positives += torch.sum(predicted & labels).item()
                false_positives += torch.sum(predicted & torch.logical_not(labels)).item()
                true_negatives += torch.sum(torch.logical_not(predicted) & torch.logical_not(labels)).item()
                false_negatives += torch.sum(torch.logical_not(predicted) & labels).item()
                scores_list.append(scores.cpu())
                label_list.append(labels.cpu())
                if len(labels) < len(recall_values_acc):
                    continue
                candidate_idx_sorted_by_score = torch.argsort(scores.squeeze(), descending=True)
                labels_sorted_by_score = labels[candidate_idx_sorted_by_score]
                cumulative = torch.cumsum(labels_sorted_by_score, dim=0)
                cutoff = int(min(self.config['top_span_ratio'] * len(labels), self.config['max_num_extracted_spans']))
                recall_actual_acc += cumulative[cutoff].item()
                step_idx = torch.arange(0.0, 1.01, 0.01, device=cumulative.device) * cumulative.shape[0]
                step_idx[-1] -= 1
                recall_values = cumulative[step_idx.to(torch.long)]
                recall_values[torch.isnan(recall_values)] = 0.0
                recall_values_acc += recall_values

        precision = true_positives / (true_positives + false_positives) if true_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        recall_values_acc /= positive_examples
        recall_values_acc = recall_values_acc.cpu()
        logger.info(recall_values_acc.tolist())
        logger.info(total / len(tensor_examples))
        metrics = {
            'precision': precision,
            'recall': recall,
            'accuracy': (true_positives + true_negatives) / total, # This is pretty useless here due to the number of true negatives
            'f1': f1,
            'recalls': recall_actual_acc / positive_examples
        }
        if tb_writer:
            tb_writer.add_pr_curve(
                "mention_pr",
                labels=torch.cat(label_list),
                predictions=torch.cat(scores_list).T.squeeze(),
                num_thresholds=1000,
                global_step=step,
            )
            tb_writer.add_histogram('relative recall', recall_values_acc, i, bins=len(recall_values_acc))
            name = f'{self.name_suffix}_pr_{step}.csv'
            out_file = open(self.config['log_dir'] + '/' + name, 'w')
            writer = csv.writer(out_file)
            writer.writerow(['precision', 'recall', 'threshold'])
            curve = sklearn.metrics.precision_recall_curve(
                torch.cat(label_list),
                torch.cat(scores_list).T.squeeze(),
            )
            for (precision, recall, thresh) in zip(*curve):
                writer.writerow([precision, recall, thresh])
            out_file.close()
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)
        logger.info('Positive weight for this distribution: %.5f' % (negative_examples / positive_examples))

        subtoken_map = stored_info['subtoken_maps']
        duplicates_removed = {}
        # remove duplicates
        for doc_key, clusters in doc_to_prediction.items():
            handled = set()
            doc_no_duplicates = []
            for cluster_id, mentions in enumerate(clusters):
                new_mentions = []
                for start, end in mentions:
                    start_word, end_word = subtoken_map[doc_key][start], subtoken_map[doc_key][end]
                    if (start_word, end_word) in handled:
                        pass
                    else:
                        handled.add((start_word, end_word))
                        new_mentions.append((start, end))
                doc_no_duplicates.append(new_mentions)
            duplicates_removed[doc_key] = doc_no_duplicates

        if official:
            conll_results = conll.evaluate_conll(
                self.config['conll_scorer'],
                conll_path,
                duplicates_removed,
                stored_info['subtoken_maps'],
                out_file,
                merge_overlapping_spans=self.config['postprocess_merge_overlapping_spans'],
            )
            official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            logger.info('Official avg F1: %.4f' % official_f1)
        return f1, metrics

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

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_task),
            LambdaLR(optimizers[1], lr_lambda_task),
        ]
        return schedulers
        # return LambdaLR(optimizer, [lr_lambda_bert, lr_lambda_bert, lr_lambda_task, lr_lambda_task])

    def save_model_checkpoint(self, model, step):
        suffix = f'{self.name_suffix}_{step}'
        self.old_save_suffix = self.last_save_suffix
        self.last_save_suffix = suffix
        path_ckpt = join(self.config['log_dir'], f"model_{self.last_save_suffix}.bin")
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, model, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)

    def delete_old_checkpoint(self):
        if self.old_save_suffix is None or ('keep_all_saved_models' in self.config \
            and self.config['keep_all_saved_models'] == True):
            return

        path_ckpt = join(self.config['log_dir'], f'model_{self.old_save_suffix}.bin')
        remove(path_ckpt)

def run_mentions(config_name, gpu_id):
    runner = MentionRunner(config_name, gpu_id)
    model = runner.initialize_model()

    runner.train(model)

    # Restore best parameters
    runner.load_model_checkpoint(model, runner.last_save_suffix)

    stored_info = runner.data.get_stored_info()
    examples_train, examples_dev, examples_test = runner.data.get_tensor_examples()

    path_dev_pred = join(runner.config['log_dir'], f'dev_{runner.last_save_suffix}.prediction')
    path_test_pred = join(runner.config['log_dir'], f'test_{runner.last_save_suffix}.prediction')
    runner.evaluate(model, examples_dev, stored_info, 0, official=True, conll_path=runner.config['conll_eval_path'], out_file=path_dev_pred)
    runner.evaluate(model, examples_test, stored_info, 0, official=True, conll_path=runner.config['conll_test_path'], out_file=path_test_pred)
    return model, runner

if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    run_mentions(config_name, gpu_id)
