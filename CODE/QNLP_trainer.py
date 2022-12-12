

import sys
import torch


import pandas as pd
from lambeq import BobcatParser
from discopy import Dim
from lambeq import AtomicType, SpiderAnsatz
from lambeq import PytorchModel
from lambeq import PytorchTrainer
from lambeq import Dataset
import matplotlib.pyplot as plt

def preparedata(loc):
    labels, sentences = [], []
    with open(loc) as f:
        for line in f:
            t = float(line[0])
            labels.append([t, 1-t])
            line_l = line.split()[1:]
            line_l = [x.split("_")[0]for x in line_l]
            line_final = " ".join(line_l)
            sentences.append(line_final)
            
    return labels, sentences


train_labels, train_data = preparedata("qnlp_lorenz_etal_2021_resources-main/datasets/mc_train_data.txt")
val_labels, val_data = preparedata("qnlp_lorenz_etal_2021_resources-main/datasets/mc_dev_data.txt")
test_labels, test_data = preparedata("qnlp_lorenz_etal_2021_resources-main/datasets/mc_test_data.txt")
val_data = val_data[:28] + val_data[29:]
val_labels = val_labels[:28] + val_labels[29:]

print(len(val_labels))
print(len(val_data))
print(val_labels)


"""
#converting the sentences into diagrams
"""



parser = BobcatParser(verbose='text')

train_diagrams = parser.sentences2diagrams(train_data)
val_diagrams = parser.sentences2diagrams(val_data)
test_diagrams = parser.sentences2diagrams(test_data)

"""
converting the diagrams into circuit
"""


ansatz = SpiderAnsatz({AtomicType.NOUN: Dim(2),
                       AtomicType.SENTENCE: Dim(2)})

train_circuits = [ansatz(diagram) for diagram in train_diagrams]
val_circuits =  [ansatz(diagram) for diagram in val_diagrams]
test_circuits = [ansatz(diagram) for diagram in test_diagrams]

#train_circuits[2].draw()
#val_circuits[8].draw()

"""
Initialize model using the circuit diagrams
"""

all_circuits = train_circuits + val_circuits + test_circuits
model = PytorchModel.from_diagrams(all_circuits)

"""
Define accuracy as a metrics of choice
"""

sig = torch.sigmoid

def accuracy(y_hat, y):
    return torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2  # half due to double-counting

eval_metrics = {"acc": accuracy}

"""
Initilize pytorch trainer
"""


BATCH_SIZE = 30
EPOCHS = 40
LEARNING_RATE = 3e-2
SEED = 0

trainer = PytorchTrainer(
        model=model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose='text',
        seed=SEED)


"""
Prepare dataset for training and validation
"""

train_dataset = Dataset(
            train_circuits,
            train_labels,
            batch_size=BATCH_SIZE)

val_dataset = Dataset(val_circuits, val_labels, shuffle=False)



"""
Fit the model on the dataset
"""
#trainer.fit(train_dataset, val_dataset, evaluation_step=1, logging_step=5)



def plotLossAndAccuracy(trainer, name):
    fig1, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharey='row', figsize=(10, 6))
    ax_tl.set_title('Training set')
    ax_tr.set_title('Development set')
    ax_bl.set_xlabel('Epochs')
    ax_br.set_xlabel('Epochs')
    ax_bl.set_ylabel('Accuracy')
    ax_tl.set_ylabel('Loss')

    colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    ax_tl.plot(trainer.train_epoch_costs, color=next(colours))
    ax_bl.plot(trainer.train_results['acc'], color=next(colours))
    ax_tr.plot(trainer.val_costs, color=next(colours))
    ax_br.plot(trainer.val_results['acc'], color=next(colours))
    name = name + ".png"
    plt.savefig(name)
    


#plotLossAndAccuracy(trainer, "defaultmodel")
# print test accuracy
# test_acc = accuracy(model(test_circuits), torch.tensor(test_labels))
# print('Test accuracy:', test_acc.item())


#sys.exit()
class MyCustomModel(PytorchModel):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(2, 4)
        #self.relu = torch.nn.ReLU()
        #self.dropout = torch.nn.Dropout(0.2)
        self.net = torch.nn.Linear(4, 2)
        #self.net = torch.nn.Linear(2,2)

    def forward(self, input):
        """define a custom forward pass here"""
        preds = self.get_diagram_output(input)
        preds = self.dense(preds)
        #preds = self.relu(preds)
        #preds = self.dropout(preds)
        preds = self.net(preds)
        return preds

custom_model = MyCustomModel.from_diagrams(all_circuits)
custom_model_trainer = PytorchTrainer(
        model=custom_model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose='text',
        seed=SEED)
custom_model_trainer.fit(train_dataset, val_dataset, logging_step=5)

plotLossAndAccuracy(custom_model_trainer, "CustomModelSigmoid")

test_acc = accuracy(custom_model(test_circuits), torch.tensor(test_labels))
print('Test accuracy:', test_acc.item())