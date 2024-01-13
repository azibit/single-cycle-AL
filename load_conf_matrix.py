import pickle, math, random
import glob as glob

# Load the confusion matrix from the saved file
with open('confusion_matrix_pt4al_test_sample.pkl', 'rb') as file:
    confusion = pickle.load(file)

paths = glob.glob('./DATA-cifar_sub_100' + '/train/*/*')
index_value = random.randint(0, 999)

print("Index to check: ", index_value)
print("File Path: ", )
print("Description: ", paths[index_value], "of class:", paths[index_value].split("/")[-2])

# Access the index_value row of the confusion matrix
numbers = confusion[index_value]

print("Total Number: ", len(numbers))
print("Sum: ", sum(numbers))

# Print the first row
# print("First Row of Confusion Matrix:")
# print(first_row)

# Remove a certain value (e.g., remove all occurrences of 4)
# index_value = 102
filtered_numbers = [numbers[i] for i in range(len(numbers)) if i != index_value]

# Calculate the average of the numbers
average = math.floor((sum(filtered_numbers) / len(filtered_numbers)) + 0.5)

# Find indices greater than the average
indices_greater_than_average = [i for i, x in enumerate(filtered_numbers) if x > average]

# Print results
print("Average:", average)
# print("Filtered List (without value {}):\n".format(index_value), filtered_numbers)
print("Indices Greater Than Average:\n", indices_greater_than_average)

max_value = max(filtered_numbers)
print("Max Value: ", max_value)

# Find indices greater than the average
indices_at_max = [i for i, x in enumerate(filtered_numbers) if x == max_value]
print("Indices at Max Value: ", [(paths[i] + " of class: " + paths[i].split("/")[-2]) for i in indices_at_max])

# import pickle

# # Load the confusion matrix from the saved file
# with open('confusion_matrix_pt4al_test_sample.pkl', 'rb') as file:
#     confusion = pickle.load(file)
# print(confusion)
# # for i in range(len(confusion)):
# #     print("At index", i, "The sum is", sum(confusion[i][:]))

# import numpy as np
# from sklearn.metrics import confusion_matrix

# total_truths = np.loadtxt('total_truths_cifar10_100_per_class')
# total_preds = np.loadtxt('total_preds_cifar10_100_per_class')

# confusion = confusion_matrix(total_truths, total_preds)
# for i in range(len(confusion)):
#     print("At index", i, "The sum is", sum(confusion[i][:]))