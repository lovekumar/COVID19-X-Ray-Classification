import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score,precision_score


def plot_confusion_matrix(results, class_names, plt_file='confusion_matrix.png'):
	'''
	This function is used for storing Normalized Confusion Matrix
	This code is referred from
		https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html

	:param results: list of tuple ("label", "prediction")
	:param class_names: prediction classes like (covid,normal,Pneumonia)
	:param plt_file: output file name for storing confusion matrix
	:return: None
	'''

	y_true, y_pred = zip(*results)

	conf_matrix = confusion_matrix(y_true, y_pred)
	conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
	print(conf_matrix)
	plt.figure(figsize=(8,8))
	plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Normalized Confusion Matrix')
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=30)
	plt.yticks(tick_marks, class_names)

	fmt = '.2f'
	thresh = conf_matrix.max() / 2.
	for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
		plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center", color="white" if conf_matrix[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True')
	plt.xlabel('Predicted')

	plt.subplots_adjust(bottom=0.18,left=0.18)
	plt.savefig(plt_file)


def evaluate_model(results):
	'''
	This function is used for calculating accuracy_score, f1_score, recall_score, precision_score

	:param results: list of tuple ("label", "prediction")
	:return: accuracy_score, f1_score, recall_score, precision_score
	'''
	y_true, y_pred = zip(*results)

	acc = accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred, average='micro')
	recall = recall_score(y_true, y_pred, average='micro')
	precision = precision_score(y_true, y_pred, average='micro')

	return acc, f1, recall, precision