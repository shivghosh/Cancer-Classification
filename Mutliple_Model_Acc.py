
import tensorflow as tf
from keras.models import load_model
import numpy as np
import os
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
import matplotlib.pyplot as plt
import pandas as pd


# classifications = ['colon_aca','colon_n','lung_aca','lung_n','lung_scc']
image_dir_path = os.path.join('lung_colon_image_set','lung')

classifications = os.listdir(image_dir_path)
batch_size = 32

# Load the data
data = tf.keras.utils.image_dataset_from_directory(image_dir_path,
                                                    class_names=classifications,
                                                    shuffle=True,
                                                    color_mode='rgb',
                                                    batch_size=batch_size,labels='inferred')

data_iterator = data.as_numpy_iterator()

train_size = int(len(data)*.7)
test_size = int(len(data)*.3)

train = data.take(train_size)
test = data.skip(train_size).take(test_size)



accuracy_list = []
precision_list = []
recall_list = []

import os
for root, dirs, files in os.walk("PastModels", topdown=False):
    for name in files:
        print("+"*50)
        print("Model: ",name)
        print("+"*50)
        model = load_model(os.path.join(root, name))


        acc = CategoricalAccuracy()
        pre = Precision()
        re = Recall()
        for imgs, ys in test:
            # Make a prediction for the entire batch
            yhats = model.predict(imgs,verbose=0)
            # Convert the predictions to class indices
            yhats = np.argmax(yhats, axis=-1)
            # Update the metrics
            acc.update_state(ys, yhats)
            pre.update_state(ys, yhats)
            re.update_state(ys, yhats)
        print(f'Accuracy: {acc.result().numpy()}')
        print(f'Precision: {pre.result().numpy()}')
        print(f'Recall: {re.result().numpy()}')

        accuracy_list.append(acc.result().numpy())
        precision_list.append(pre.result().numpy())
        recall_list.append(re.result().numpy())
        print("*"*50)
    
    import matplotlib.pyplot as plt

    # Assuming accuracy_list, precision_list, recall_list contain the scores,
    # and files contain the model names

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))  # 3 Rows, 1 Column


    # Plot bar plots
    axs[0].bar(files, accuracy_list, color='skyblue', edgecolor='black')
    axs[0].set_title('Accuracy Bar Plot')
    axs[0].set_xlabel('Models')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xticklabels(files, rotation=45, ha="right")

    axs[1].bar(files, precision_list, color='lightgreen', edgecolor='black')
    axs[1].set_title('Precision Bar Plot')
    axs[1].set_xlabel('Models')
    axs[1].set_ylabel('Precision')
    axs[1].set_xticklabels(files, rotation=45, ha="right")

    axs[2].bar(files, recall_list, color='salmon', edgecolor='black')
    axs[2].set_title('Recall Bar Plot')
    axs[2].set_xlabel('Models')
    axs[2].set_ylabel('Recall')
    axs[2].set_xticklabels(files, rotation=45, ha="right")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.savefig("Model_Metrics.png")
