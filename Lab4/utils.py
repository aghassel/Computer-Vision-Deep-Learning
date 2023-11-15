import matplotlib.pyplot as plt

def plot_loss(train_loss, val_loss, save_path):
    plt.figure()
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_path)
