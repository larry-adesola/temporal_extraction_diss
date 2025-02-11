from torch.utils.data import Subset
from data_utils.data import temprel_set


def contextualise_data(tokeniser, trainsetLoc, testsetLoc, model):

    traindevset = temprel_set(trainsetLoc)
    traindev_tensorset = traindevset.to_tensor(tokenizer=tokeniser, pos_enabled=model >= 4)
    if "temprel" not in trainsetLoc:
      total_len = len(traindev_tensorset)
      split_point = int(0.9 * total_len)  # 90% for training, 10% for dev
      train_idx = list(range(split_point))
      dev_idx = list(range(split_point, total_len))
    else:
      train_idx = list(range(len(traindev_tensorset)-1852))
      dev_idx = list(range(len(traindev_tensorset)-1852, len(traindev_tensorset)))
    train_tensorset = Subset(traindev_tensorset, train_idx)
    dev_tensorset = Subset(traindev_tensorset, dev_idx) #Last 21 docs

    testset = temprel_set(testsetLoc)
    test_tensorset = testset.to_tensor(tokenizer=tokeniser, pos_enabled=model >= 4)
    return train_tensorset, dev_tensorset, test_tensorset