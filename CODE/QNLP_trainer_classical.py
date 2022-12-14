


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
from sklearn.model_selection import train_test_split



# def preparedata(loc):
#     labels, sentences = [], []
#     with open(loc) as f:
#         for line in f:
#             t = float(line[0])
#             labels.append([t, 1-t])
#             line_l = line.split()[1:]
#             line_l = [x.split("_")[0]for x in line_l]
#             line_final = " ".join(line_l)
#             sentences.append(line_final)
            
#     return labels, sentences


# train_labels, train_data = preparedata("qnlp_lorenz_etal_2021_resources-main/datasets/mc_train_data.txt")
# val_labels, val_data = preparedata("qnlp_lorenz_etal_2021_resources-main/datasets/mc_dev_data.txt")
# test_labels, test_data = preparedata("qnlp_lorenz_etal_2021_resources-main/datasets/mc_test_data.txt")
# val_data = val_data[:28] + val_data[29:]
# val_labels = val_labels[:28] + val_labels[29:]

# print(len(val_labels))
# print(len(val_data))
# print(val_labels)


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

#print(len(train_data_all))
#print(len(train_labels_all))
#sys.exit()
X_train, test_data, X_test, test_labels = train_test_split(train_data_all, train_labels_all, test_size=0.15, random_state=42)


train_data, val_data, train_labels, val_labels = train_test_split(X_train, X_test, test_size=0.15, random_state=42)

"""
#converting the sentences into diagrams
"""
#print(len(train_data))
#print(len(train_labels))
#print(len(val_labels))
#print(len(test_labels))


parser = BobcatParser(root_cats=('NP', 'N'), verbose='text')
raw_train_diagrams = parser.sentences2diagrams(train_data, suppress_exceptions=True)
raw_val_diagrams = parser.sentences2diagrams(val_data, suppress_exceptions=True)
raw_test_diagrams = parser.sentences2diagrams(test_data, suppress_exceptions=True)
#test_diagrams = parser.sentences2diagrams(test_data)


"""
converting the diagrams into circuit
"""

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

#print(train_diagrams[2].draw())

ansatz = SpiderAnsatz({AtomicType.NOUN: Dim(2),
                       AtomicType.SENTENCE: Dim(2)})

train_circuits = [ansatz(diagram) for diagram in train_diagrams]
val_circuits =  [ansatz(diagram) for diagram in val_diagrams]
test_circuits =  [ansatz(diagram) for diagram in test_diagrams]
#test_circuits = [ansatz(diagram) for diagram in test_diagrams]

#print(train_circuits[2].draw())
#val_circuits[8].draw()

"""
Initialize model using the circuit diagrams
"""

#all_circuits = train_circuits + val_circuits + test_circuits
all_circuits = train_circuits + val_circuits + test_circuits
#model = PytorchModel.from_diagrams(all_circuits)

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
EPOCHS = 70
LEARNING_RATE = 1e-3
SEED = 0

# trainer = PytorchTrainer(
#         model=model,
#         loss_function=torch.nn.BCEWithLogitsLoss(),
#         optimizer=torch.optim.AdamW,
#         learning_rate=LEARNING_RATE,
#         epochs=EPOCHS,
#         evaluate_functions=eval_metrics,
#         evaluate_on_train=True,
#         verbose='text',
#         seed=SEED)


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
        self.dense = torch.nn.Linear(2, 36)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.net = torch.nn.Linear(36, 2)
        #self.net = torch.nn.Linear(2,2)

    def forward(self, input):
        """define a custom forward pass here"""
        preds = self.get_diagram_output(input)
        preds = self.dense(preds)
        preds = self.relu(preds)
        preds = self.dropout(preds)
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

plotLossAndAccuracy(custom_model_trainer, "DenseLayer_loss_accuracy_plot_classical_36units")

test_acc = accuracy(custom_model(test_circuits), torch.tensor(test_labels))
print('Test accuracy:', test_acc.item())
