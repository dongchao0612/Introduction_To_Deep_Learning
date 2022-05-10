import numpy as np

from mypackge.mnist import load_mnist

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    train_size = x_train.shape[0]
    print(train_size)
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)  # 随机选择0-60000之间的10个数字
    print(batch_mask.shape)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print(t_batch.shape)
