from Library import *

def plot_utlis(pred, valid_labels, score, model_name):
  model_name = 'MLP'
  for i in range(0, 20):
    tmp = i * 10
    plt.scatter(pred[tmp][0], pred[tmp][1])
    plt.scatter(valid_labels[tmp][0], valid_labels[tmp][1], c='#407f4b')
    plt.title('distribution')
    plt.xlim(0.2, 0.8)
    plt.ylim(0.2, 0.8)
    plt.xlabel('xpts')
    plt.ylabel('ypts')
    plt.savefig(f"data/distribution/{model_name}_{i}distribution.png")
    plt.clf()


def plot_loss(history, score, i, model_name):
    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'valid'], loc='upper left')
    # plt.show()
    plt.savefig(f"data/loss/{model_name}{i}_peopleleave{score[1]}_loss.png")
    plt.clf()

def plot_distribution(preds, true):
    i = 0
