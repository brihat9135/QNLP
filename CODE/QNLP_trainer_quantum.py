
import sys
import torch


import pandas as pd
import numpy as np
from lambeq import BobcatParser
from discopy import Dim
from lambeq import AtomicType, SpiderAnsatz, IQPAnsatz, remove_cups
from lambeq import PytorchModel
from lambeq import PytorchTrainer
from lambeq import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pytket.extensions.qiskit import AerBackend, IBMQBackend
from lambeq import TketModel
from lambeq import QuantumTrainer, SPSAOptimizer



"""
Function to read data, clean the underscore in the text
"""
def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        count = 1
        for line in f:
            line_list = line.split()
            label = line_list[0]
            sentence_list = line_list[2:]
            sentence_list = [x.split("_")[0] for x in sentence_list]
            if label == "OBJ":
                label = [1.0, 0.0]
            else:
                label = [0.0, 1.0]
            labels.append(label)
            sentences.append(" ".join(sentence_list))
    return labels, sentences


train_labels, train_data = read_data('RELPRON/relpron.dev')
val_labels, val_data = read_data('RELPRON/relpron.test')

train_labels_all, train_data_all = train_labels + val_labels, train_data + val_data


X_train, test_data, X_test, test_labels = train_test_split(train_data_all, train_labels_all, test_size=0.15, random_state=42)


train_data, val_data, train_labels, val_labels = train_test_split(X_train, X_test, test_size=0.15, random_state=42)

"""
#converting the sentences into diagrams
"""



parser = BobcatParser(root_cats=('NP', 'N'), verbose='text')
raw_train_diagrams = parser.sentences2diagrams(train_data, suppress_exceptions=True)
raw_val_diagrams = parser.sentences2diagrams(val_data, suppress_exceptions=True)
raw_test_diagrams = parser.sentences2diagrams(test_data, suppress_exceptions=True)





train_diagrams = [
    diagram.normal_form()
    for diagram in raw_train_diagrams if diagram is not None
]
val_diagrams = [
    diagram.normal_form()
    for diagram in raw_val_diagrams if diagram is not None
]

test_diagrams = [
    diagram.normal_form()
    for diagram in raw_test_diagrams if diagram is not None
]

train_labels = [
    label for (diagram, label)
    in zip(raw_train_diagrams, train_labels)
    if diagram is not None]

val_labels = [
    label for (diagram, label)
    in zip(raw_val_diagrams, val_labels)
    if diagram is not None
]

test_labels = [
    label for (diagram, label)
    in zip(raw_test_diagrams, test_labels)
    if diagram is not None
]


"""
converting the diagrams into circuit
"""



ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 0},
                   n_layers=1, n_single_qubit_params=3)

train_circuits = [ansatz(remove_cups(diagram)) for diagram in train_diagrams]
val_circuits =  [ansatz(remove_cups(diagram))  for diagram in val_diagrams]
test_circuits =  [ansatz(remove_cups(diagram))  for diagram in test_diagrams]


"""
Initialize model using the circuit diagrams
"""


all_circuits = train_circuits + val_circuits + test_circuits


backend = AerBackend()

backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 512
}


model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)

"""
Define accuracy as a metrics of choice
"""

loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)  # binary cross-entropy loss

#accuracy = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting
#eval_metrics = {"acc": acc}
sig = torch.sigmoid

def accuracy(y_hat, y):
    return torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2  # half due to double-counting

eval_metrics = {"acc": accuracy}


BATCH_SIZE = 30
EPOCHS = 15
SEED = 2

"""
Initilize pytorch trainer
"""

trainer = QuantumTrainer(
    model,
    loss_function=loss,
    epochs=EPOCHS,
    optimizer=SPSAOptimizer,
    optim_hyperparams={'a': 0.05, 'c': 0.06, 'A':0.01*EPOCHS},
    evaluate_functions=eval_metrics,
    evaluate_on_train=True,
    verbose = 'text',
    seed=0
)



"""
Prepare dataset for training and validation
"""

train_dataset = Dataset(
            train_circuits,
            train_labels,
            batch_size=BATCH_SIZE)

val_dataset = Dataset(val_circuits, val_labels, shuffle=False)
test_dataset = Dataset(test_circuits, test_labels, shuffle=False)


"""
Fit the model on the dataset
"""

#print("Fitting the model")
trainer.fit(train_dataset, val_dataset, evaluation_step=1, logging_step=1)



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
    

plotLossAndAccuracy(trainer, "Quantum_loss_accuracy_plot")

test_acc = accuracy(model(test_circuits), torch.tensor(test_labels))
print('Test accuracy:', test_acc.item())
