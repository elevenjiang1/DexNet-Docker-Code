"""
Helper classes for analyzing machine learning output
Author: Jeff Mahler
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics as sm

class ConfusionMatrix(object):
	""" Confusion Matrix for classification errors """
	
	def __init__(self, num_categories):
		""" 
		Parameters
		----------
		num_categories : int
			number of prediction categories
		"""
		self.num_categories = num_categories

		# organized as row = true, column = pred
		self.matrix = np.zeros([num_categories, num_categories])

	def update(self, predictions, labels):
		""" Update the Confusion Matrix with predictions and labels

		Parameters
		----------
		predictions :obj: list
			predictions to put in the matrix
		labels :obj: list
			labels to put in the matrix
		"""
		num_pred = predictions.shape[0]
		for i in range(num_pred):
			self.matrix[labels[i].astype(np.uint16), predictions[i].astype(np.uint16)] += 1

class ClassificationResult(object):
	""" Wrapper for machine learning classification results """
	
	def __init__(self, pred_probs_list, labels_list):
		"""
		Parameters
		----------
		pred_probs_list :obj: list
			list of predicted probabilities
		labels_list :obj: list
			list of labels corresponding to predicted probabilites
		"""
		self.pred_probs = None
		self.labels = None
		
		for pred_probs, labels in zip(pred_probs_list, labels_list):
			if self.pred_probs is None:
				self.pred_probs = pred_probs
				self.labels = labels
			else:
				self.pred_probs = np.r_[self.pred_probs, pred_probs]
				self.labels = np.r_[self.labels, labels]

	@property
	def error_rate(self):
		""" Get the error rate
		
		Returns
		-------
		: float
			the error rate
		"""
		return 100.0 - (
			100.0 *
			np.sum(self.predictions == self.labels) /
			self.num_datapoints)

	def top_k_error_rate(self, k):
		""" Get the top k error rates
		
		Parameters
		----------
		k : int
			number of error rates to get

		Returns
		-------
		:obj: ndarray
			ndarray of top k error rates
		"""
		predictions_arr = self.top_k_predictions(k)
		labels_arr = np.zeros(predictions_arr.shape)
		for i in range(k):
			labels_arr[:,i] = self.labels

		return 100.0 - (
			100.0 *
			np.sum(predictions_arr == labels_arr) /
			self.num_datapoints)

	@property
	def fpr(self):
		""" Get the false positive rate 
		
		Returns
		-------
		: float
			the false positive rate 
		"""
		if np.sum(self.labels == 0) == 0:
			return 0.0
		return float(np.sum((self.predictions == 1) & (self.labels == 0))) / np.sum(self.labels == 0)

	@property
	def precision(self):
		""" Get the precision 
		
		Returns
		-------
		: float 
			the precision
		"""
		if np.sum(self.predictions == 1) == 0:
			return 1.0
		return float(np.sum((self.predictions == 1) & (self.labels == 1))) / np.sum(self.predictions == 1)

	@property
	def recall(self):
		""" Get the recall

		Returns
		------
		:float
			the recall
		"""
		if np.sum(self.predictions == 1) == 0:
			return 1.0
		try:
			return float(np.sum((self.predictions == 1) & (self.labels == 1))) / np.sum(self.labels == 1)
		except ZeroDivisionError:
			return 0.0		

	@property
	def num_datapoints(self):
		""" Get the number of datapoints

		Returns
		-------
		: int
			the number of datapoints
		"""
		return self.pred_probs.shape[0]

	@property
	def num_categories(self):
		""" Get the number of categories
		
		Returns
		-------
		: int
			the number of categories
		"""
		return self.pred_probs.shape[1]
		
	@property
	def predictions(self):
		""" Get the final predicted probability for each datapoint

		Returns
		------
		:obj: ndarray
			ndarray of final probabilities for each datapoint
		"""
		return np.argmax(self.pred_probs, 1)

	def mispredicted_indices(self):
		""" Return the set of indices mis-predicted

		Returns
		------
		:obj: ndarray
			ndarray of the mispredicted indices
		"""    	
		return np.where(self.predictions != self.labels)[0]

	def correct_indices(self):
		""" Return the set of indices correctly predicted

		Returns
		------
		:obj: ndarray
			ndarray of the correctly predicted indices

		"""
		return np.where(self.predictions == self.labels)[0]
	
	def top_k_predictions(self, k):
		""" Get the top k predictions 

		Parameters
		----------
		k : int
			number of predictions to get

		Returns
		-------
		:obj: ndarray
			ndarray of top k predictions rates
		"""
		return np.argpartition(self.pred_probs, -k, axis=1)[:, -k:]

	@property
	def confusion_matrix(self):
		""" Get a confusion matrix representing the predictions and labels in this ClassificationResult
		
		Returns
		-------
		:obj: ConfusionMatrix
			ConfusionMatrix representing the predictions and labels
		"""
		cm = ConfusionMatrix(self.num_categories)
		cm.update(self.predictions, self.labels)
		return cm

	def convert_labels(self, mapping):
		"""  Converts this ClassificationResult to a new ClassificationResult with different labels using the specified mapping

		Returns
		------
		:obj: ClassificationResult
			ClassificationResult containing new labels
		"""
		new_num_categories = len(set(mapping.values()))
		new_probs = np.zeros([self.num_datapoints, new_num_categories])
		new_labels = np.zeros(self.num_datapoints)
		for i in range(self.num_datapoints):
			for j in range(self.num_categories):
				new_probs[i,mapping[j]] += self.pred_probs[i,j]
			new_labels[i] = mapping[self.labels[i]]
		return ClassificationResult([new_probs], [new_labels])

	def label_vectors(self):
		""" Get label vectors for this ClassificationResult 
		
		Returns
		-------
		:obj: ndarray
			ndarray of predicted probabilities
		:obj: ndarray
			ndarray of labels
		"""
		return self.pred_probs[:,1], self.labels

	def multiclass_label_vectors(self):
		""" Get multiclass label vectors for this ClassificationResult

		Returns
		-------
		:obj: ndarray
			ndarray of predicted probabilities
		:obj: ndarray
			ndarray of labels
		"""
		label_mat = np.zeros(self.pred_probs.shape)
		for i in range(self.num_datapoints):
			label_mat[i, self.labels[i]] = 1

		pred_probs_vec = self.pred_probs.ravel()
		labels_vec = label_mat.ravel()
		return pred_probs_vec, labels_vec

	def precision_recall_curve(self, plot=False, line_width=2, font_size=15, color='b', style='-', label='', marker=None):
		""" Generates precision recall curve for the predictions and labels in this ClassificationResult. Optionally plots the 
		curve if the plot flag is on

		Parameters
		----------
		plot : boolean
			whether or not to plot the curve
		line_width : float
			line width to use if plotting
		font_size : int 
			font size to use if plotting
		color :obj: str
			color to use if plotting
		style :obj: str
			style to use if plotting
		label :obj: str
			label to use if plotting
		marker :obj: str
			valid marker style to use if plotting

		Returns
		-------
		precision :obj: ndarray
			ndarray of precision values
		recall :obj: ndarray
			ndarray of recall values
		thresholds :obj: ndarray
			ndarray of thresholds
		"""
		pred_probs_vec, labels_vec = self.label_vectors()
		precision, recall, thresholds = sm.precision_recall_curve(labels_vec, pred_probs_vec)
		if plot:
			plt.plot(recall, precision, linewidth=line_width, color=color, linestyle=style, label=label, marker=marker)
			plt.xlabel('Recall', fontsize=font_size)
			plt.ylabel('Precision', fontsize=font_size)
		return precision, recall, thresholds

	def roc_curve(self, plot=False, line_width=2, font_size=15, color='b', style='-', label=''):
		""" Generates receiver operating characteristic curve for the predictions and labels in this ClassificationResult. Optionally plots
		the curve if the plot flag is on

		Parameters
		----------
		plot : boolean
			whether or not to plot the curve
		line_width : float
			line width to use if plotting
		font_size : int 
			font size to use if plotting
		color :obj: str
			color to use if plotting
		style :obj: str
			style to use if plotting
		label :obj: str
			label to use if plotting

		Returns
		-------
		fpr :obj: ndarray
			false positive rates
		tpr :obj: ndarray
			true positive rates
		thresholds :obj: ndarray
			thresholds
		"""
		pred_probs_vec, labels_vec = self.multiclass_label_vectors()
		fpr, tpr, thresholds = sm.roc_curve(labels_vec, pred_probs_vec)

		if plot:
			plt.plot(fpr, tpr, linewidth=line_width, color=color, linestyle=style, label=label)
			plt.xlabel('FPR', fontsize=font_size)
			plt.ylabel('TPR', fontsize=font_size)
		return fpr, tpr, thresholds

	@property
	def ap_score(self):
		""" Get the average precision score for this ClassificationResult

		Returns
		-------
		: float
			the average precision score
		"""
		pred_probs_vec, labels_vec = self.multiclass_label_vectors()
		return sm.average_precision_score(labels_vec, pred_probs_vec)        

	@property
	def auc_score(self):
		""" Get the area under curve score for this ClassificationResult
		
		Returns
		-------
		: float
			the area under curve score
		"""
		pred_probs_vec, labels_vec = self.multiclass_label_vectors()
		return sm.roc_auc_score(labels_vec, pred_probs_vec)        

	def save(self, filename):
		""" Saves the predictions and labels stored in this ClassificationResult to respective npz files in the directory 
		specified by filename
		
		Parameters
		----------
		filename :obj: str
			the directory to which to store the predictions and labels
		"""
		if not os.path.exists(filename):
			os.mkdir(filename)
		
		pred_filename = os.path.join(filename, 'predictions.npz')
		np.savez_compressed(pred_filename, self.pred_probs)

		labels_filename = os.path.join(filename, 'labels.npz')
		np.savez_compressed(labels_filename, self.labels)

	@staticmethod
	def load(filename):
		""" Loads the predictions and labels stored in `predictions.npz` and `labels.npz` in the directory specified by
		filename into a ClassificationResult
		
		Parameters
		----------
		filename :obj: str
			the directory from which to load the predictions and labels

		Returns
		-------
		:obj: ClassificationResult
			a ClassificationResult object containing the predictions and labels
		"""
		if not os.path.exists(filename):
			raise ValueError('File %s does not exists' %(filename))

		pred_filename = os.path.join(filename, 'predictions.npz')
		pred_probs = np.load(pred_filename)['arr_0']

		labels_filename = os.path.join(filename, 'labels.npz')
		labels = np.load(labels_filename)['arr_0']
		return ClassificationResult([pred_probs], [labels])

class RegressionResult(object):
	""" Wrapper for machine learning regression results """

	def __init__(self, predictions_list, labels_list):
		"""
		Parameters
		----------
		predictions_list :obj: list
			list of predictions
		labels_list :obj: list
			list of labels
		"""
		self.predictions = None
		self.labels = None
		
		for predictions, labels in zip(predictions_list, labels_list):
			if self.predictions is None:
				self.predictions = predictions
				self.labels = labels
			else:
				self.predictions = np.r_[self.predictions, predictions]
				self.labels = np.r_[self.labels, labels]

	@property
	def error_rate(self):
		""" Get the error rate
		
		Returns
		-------
		: float
			the error rate
		"""
		return np.sum((self.predictions - self.labels)**2) / (float(self.num_datapoints  * self.predictions.shape[1]))

	@property
	def num_datapoints(self):
		""" Get the number of datapoints

		Returns
		-------
		: int
			the number of datapoints
		"""
		return self.predictions.shape[0]

	def save(self, filename):
		""" Saves the predictions and labels stored in this RegressionResult to respective npz files in the directory 
		specified by filename
		
		Parameters
		----------
		filename :obj: str
			the directory to which to store the predictions and labels
		"""
		if not os.path.exists(filename):
			os.mkdir(filename)
		
		pred_filename = os.path.join(filename, 'predictions.npz')
		np.savez_compressed(pred_filename, self.predictions)

		labels_filename = os.path.join(filename, 'labels.npz')
		np.savez_compressed(labels_filename, self.labels)

	@staticmethod
	def load(filename):
		""" Loads the predictions and labels stored in `predictions.npz` and `labels.npz` in the directory specified by
		filename into a RegressionResult
		
		Parameters
		----------
		filename :obj: str
			the directory from which to load the predictions and labels

		Returns
		-------
		:obj: RegressionResult
			a RegressionResult object containing the predictions and labels
		"""
		if not os.path.exists(filename):
			raise ValueError('File %s does not exists' %(filename))

		pred_filename = os.path.join(filename, 'predictions.npz')
		predictions = np.load(pred_filename)['arr_0']

		labels_filename = os.path.join(filename, 'labels.npz')
		labels = np.load(labels_filename)['arr_0']
		return RegressionResult([predictions], [labels])
