import matplotlib.pyplot as plt
import numpy as np


class ImageProcessing:

	def __init__(self, training_images, training_labels):
		self.training_images = training_images
		self.training_labels = training_labels

	def normalize(self, rescale):
		self.training_images = self.training_images * rescale;

	def plot(self, indexes):
		fig = plt.figure()
		rows = int(len(indexes) - (len(indexes) / 2))
		columns = int(len(indexes) - rows)
		images = self.training_images
		new_index = 0

		for i in range(0, rows):
			for j in range(0, columns):
				new_index += 1;

				fig.add_subplot(rows, columns, new_index)
				plt.axis('off')

				temp = images[indexes[new_index - 1]]
				temp = np.array(temp)
				temp = temp.reshape((28, 28))

				plt.imshow(temp)

		plt.show()



