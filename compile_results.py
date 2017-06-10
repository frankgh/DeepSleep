import os
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def plot_confusion_matrix(output_dir, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, 'conf_mat.png'), dpi=100)
    plt.clf()
    plt.cla()
    plt.close()


def plot_roc_curve(output_dir, n_classes, y_true, y_pred):
    """
    Compute ROC curve and ROC area for each class
    :param output_dir: where to save the png image file
    :param n_classes: number of classes
    :param y_true: the true values
    :param y_pred: predicted values
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_score.png'), dpi=100)
    plt.clf()
    plt.cla()
    plt.close()

    roc_score = metrics.roc_auc_score(y_true, y_pred)
    print "ROC AUC Score: ", roc_score


def plot_accuracy(output_dir, acc, val_acc):
    """
    Summarize history for accuracy
    :param output_dir: the output directory for the plot png file
    :param acc: training accuracy list
    :param val_acc: validation accuracy list
    """
    plt.plot(acc, linewidth=2)
    plt.plot(val_acc, linestyle='dotted', linewidth=2)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='bottom right')
    plt.savefig(os.path.join(output_dir, 'accuracy.png'), dpi=100)
    plt.clf()
    plt.cla()
    plt.close()


def plot_loss(output_dir, loss, val_loss):
    """
    Summarize history for loss
    :param output_dir: the output directory for the plot png file
    :param loss: training loss history
    :param val_loss: validation loss history
    """
    plt.plot(loss, linewidth=2)
    plt.plot(val_loss, linestyle='dotted', linewidth=2)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(os.path.join(output_dir, 'loss.png'), dpi=100)
    plt.clf()
    plt.cla()
    plt.close()
