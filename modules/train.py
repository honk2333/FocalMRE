import os
import torch
from torch import optim
from transformers import AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup

from .metrics import eval_result
from .BalancedDataParallel import BalancedDataParallel


class BertTrainer(object):
    def __init__(self, train_data=None, dev_data=None, test_data=None, re_dict=None, model=None, process=None,
                 args=None, logger=None, writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.re_dict = re_dict
        self.model = model
        # self.model = torch.nn.DataParallel(self.model)
        # self.model = self.model.cuda()
        # print(next(self.model.parameters()).device)

        self.process = process
        self.logger = logger
        self.writer = writer
        self.refresh_step = 2

        self.best_dev_epoch = None
        self.best_test_epoch = None

        self.best_test_metric = 0
        self.best_test_acc = 0
        self.best_dev_metric = 0
        self.best_dev_acc = 0

        self.optimizer = None

        self.step = 0
        self.args = args

        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
            self.before_multimodal_train()

    def train(self):
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        # if self.args.load_path is not None:  # load model from load_path
        #     self.logger.info("Loading model from {}".format(self.args.load_path))
        #     self.model.load_state_dict(torch.load(self.args.load_path))
        #     self.logger.info("Load model successful!")

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs + 1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (loss, logits), labels = self._step(batch, mode="train")
                    avg_loss += loss.detach().mean().cpu().item()

                    loss.sum().backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer is not None:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)  # tensorbordx
                        avg_loss = 0

                if epoch >= self.args.eval_begin_epoch:
                    # self.evaluate(epoch)  # generator to dev.
                    self.test(epoch)

            pbar.close()
            self.pbar = None
            # self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}, acc = {}".format(self.best_dev_epoch,
            #                                                                                                   self.best_dev_metric,
            #                                                                                                   self.dev_acc))
            self.logger.info("Get best test performance at epoch {}, best test f1 score is {}, acc = {}".format(self.best_test_epoch,
                                                                                                                self.best_test_metric,
                                                                                                                self.test_acc))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        step = 0
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                             batch)  # to cpu/cuda device
                    (loss, logits), labels = self._step(batch, mode="dev")  # logits: batch, 3
                    total_loss += loss.detach().mean().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = classification_report(y_true=true_labels, y_pred=pred_labels,
                                                  labels=list(self.re_dict.values())[1:],
                                                  target_names=list(self.re_dict.keys())[1:], digits=4)
                # self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_p, micro_r, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_p'] * 100, 4), round(result['micro_r'] * 100, 4), round(result['micro_f1'] * 100, 4)
                if self.writer is not None:
                    self.writer.add_scalar(tag='dev_acc', scalar_value=acc, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='dev_p', scalar_value=micro_p, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='dev_r', scalar_value=micro_r, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='dev_f1', scalar_value=micro_f1, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss / len(self.dev_data), global_step=epoch)  # tensorbordx

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, acc: {}." \
                                 .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch, micro_f1, acc))
                if micro_f1 >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = micro_f1  # update best metric(f1 score)
                    self.dev_acc = acc
                    # if self.args.save_path is not None:  # save model
                    #     torch.save(self.model.state_dict(), self.args.save_path + "/best_model.pth")
                    #     self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def save_pred(self, true_labels, pred_labels, sk_result):
        def get_key(dic, value):
            return [k for k, v in dic.items() if v == value][0]

        with open(os.path.join(self.args.save_path, 'pred.txt'), 'w', encoding="utf-8") as wf:
            for a, b in zip(true_labels, pred_labels):
                wf.write("true: " + get_key(self.re_dict, a) + "\t, pred: " + get_key(self.re_dict, b) + "\n")
            wf.write(sk_result)
            print('Successful write!!')

    def test(self, epoch):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                             batch)  # to cpu/cuda device
                    (loss, logits), labels = self._step(batch, mode="dev")  # logits: batch, 3
                    total_loss += loss.detach().mean().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())

                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = classification_report(y_true=true_labels, y_pred=pred_labels,
                                                  labels=list(self.re_dict.values())[1:],
                                                  target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_p, micro_r, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_p'] * 100, 4), round(result['micro_r'] * 100, 4), round(result['micro_f1'] * 100, 4)
                if self.writer is not None:
                    self.writer.add_scalar(tag='test_acc', scalar_value=acc, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='test_p', scalar_value=micro_p, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='test_r', scalar_value=micro_r, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='test_f1', scalar_value=micro_f1, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='test_loss', scalar_value=total_loss / len(self.test_data), global_step=epoch)  # tensorbordx
                total_loss = 0
                self.logger.info("Epoch {}/{}, best test f1: {}, best epoch: {}, current test f1 score: {}, acc: {}" \
                                 .format(epoch, self.args.num_epochs, self.best_test_metric, self.best_test_epoch,
                                         micro_f1, acc))
                if micro_f1 >= self.best_test_metric:  # this epoch get best performance
                    self.best_test_metric = micro_f1
                    self.best_test_epoch = epoch
                    self.test_acc = acc
                    self.save_pred(true_labels, pred_labels, sk_result)
                    if self.args.save_path is not None:  # save model
                        torch.save(self.model.state_dict(), self.args.save_path + "/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def predict(self):
        self.model.eval()
        self.logger.info("\n***** Running predicting *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Predicting")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    (loss, logits), labels = self._step(batch, mode="dev")  # logits: batch, 3
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())

                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = classification_report(y_true=true_labels, y_pred=pred_labels,
                                                  labels=list(self.re_dict.values())[1:],
                                                  target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                # save predict results
                import os
                def get_key(dic, value):
                    return [k for k, v in dic.items() if v == value][0]

                with open(os.path.join(self.args.save_path, 'result.txt'), 'w', encoding="utf-8") as wf:
                    for a, b in zip(true_labels, pred_labels):
                        wf.write("true: " + get_key(self.re_dict, a) + "\t, pred: " + get_key(self.re_dict, b) + "\n")
                    wf.write(sk_result)
                    print('Successful write!!')

        self.model.train()

    def _step(self, batch, mode="train"):
        input_ids, token_type_ids, attention_mask, labels, images, box_imgs, = batch
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, box_imgs=box_imgs, obj_imgs=None, img_mask=None)
        return outputs, labels

    def before_multimodal_train(self):
        optimizer_grouped_parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'model' in name:
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)

        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        # params = list(self.model.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # grouped_params = [
        #     {
        #         'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
        #         'weight_decay': 0.01,
        #         'lr': self.args.lr,
        #         'ori_lr': self.args.lr
        #     },
        #     {
        #         'params': [p for n, p in params if any(nd in n for nd in no_decay)],
        #         'weight_decay': 0.0,
        #         'lr': self.args.lr,
        #         'ori_lr': self.args.lr
        #     }
        # ]
        # self.optimizer = AdamW(grouped_params, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)
        # for name, par in self.model.named_parameters():
        #     print(name, par.requires_grad)
