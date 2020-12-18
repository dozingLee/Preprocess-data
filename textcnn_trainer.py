import torch
import torch.nn.functional as F
import os
import sys


class TextCNNTrainer:
    def __init__(self, args):
        super(TextCNNTrainer, self).__init__()
        self.lr = args.lr
        self.epoch = args.epoch
        self.log_interval = args.log_interval
        self.test_interval = args.test_interval
        self.save_best = args.save_best
        self.model_save_dir = args.model_save_dir
        self.early_stopping = args.early_stopping

    def train(self, train_iter, val_iter, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        steps = 0
        best_acc = 0
        last_step = 0
        model.train()

        for epoch in range(1, self.epoch + 1):
            for batch in train_iter:
                with torch.no_grad():
                    feature, target = batch.text.t_(), batch.label

                optimizer.zero_grad()
                logits = model(feature)
                loss = F.cross_entropy(logits, target)
                loss.backward()
                optimizer.step()

                steps += 1

                # 通过第一次batch的按顺序数据获取，并验证最后整个batch的数据存取顺序，再次验证：
                # (1) 一个batch的数据就是一个分布式计算的最小单位
                # (2) 一个batch在整个batch运行的顺序是不能确定的，就是不能用一个batch的数据顺序验证最后的结果
                # (3) 以下代码是打印第一个batch的数据，用于比较最后的结果。已经没有用了
                # if epoch == 1 and steps < 10:
                #     index = steps - 1
                #     print('\n\nsteps: ', steps, ' , label: ', target[index], ' ,feature: ', feature[index])
                #     feature_list = []
                #     for feature_item in feature[index]:
                #         feature_list.append(batch.dataset.fields['text'].vocab.itos[feature_item])
                #     print(feature_list, ' ', batch.dataset.fields['label'].vocab.itos[target[index]] + '\n\n')

                if steps % self.log_interval == 0:
                    # torch.max(logits, 1)函数：返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
                    corrects = (torch.max(logits, 1)[1] == target).sum()
                    train_acc = 100.0 * corrects / batch.batch_size
                    sys.stdout.write('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'
                                     .format(steps, loss.item(), train_acc, corrects, batch.batch_size))

                if steps % self.test_interval == 0:
                    model.eval()
                    dev_acc = self.validate(model, val_iter)
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        last_step = steps
                        if self.save_best:
                            print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                            self.save(model, self.model_save_dir, 'best', steps)
                    else:
                        if steps - last_step >= self.early_stopping:
                            print('\nearly stop by {} steps, acc: {:.4f}%'.format(self.early_stopping, best_acc))
                            raise KeyboardInterrupt
                    model.train()

    def validate(self, model, val_iter):
        corrects, avg_loss = 0, 0
        for batch in val_iter:
            with torch.no_grad():
                feature, target = batch.text.t_(), batch.label

            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            avg_loss += loss.item()
            corrects += (torch.max(logits, 1)
                         [1].view(target.size()) == target).sum()
        size = len(val_iter.dataset)
        avg_loss /= size
        accuracy = 100.0 * corrects / size
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'
              .format(avg_loss, accuracy, corrects, size))
        return accuracy

    def save(self, model, save_dir, save_prefix, steps):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_prefix = os.path.join(save_dir, save_prefix)
        # save procedure model
        save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
        torch.save(model.state_dict(), save_path)
        # save best model
        save_best_path = '{}_steps.pt'.format(save_prefix)
        torch.save(model.state_dict(), save_best_path)

    def test(self, model, test_iter):
        model.load_state_dict(torch.load('./{}/best_steps.pt'.format(self.model_save_dir)))
        model.eval()
        self.validate(model, test_iter)
