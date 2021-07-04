import numpy as np


from DataLoader import load_data, generate_data
from Model import LogisticRegression, SVM
import Functions
import Optimizer



# 1. Prepare datasets
train_feature, train_label, test_feature, test_label = load_data("ijcnn1")
#train_feature, train_label, test_feature, test_label = generate_data([-1, -1], [1, 1], 0.5, 0.5, 50000, 50000)


# 2. Set up
epochs = 500
learning_rate = 0.001

#model = LogisticRegression(lmd=0.1)
model = SVM(lmd=0.1, delta=0.001)
model.param_initializer(train_feature.shape[1] + 1, initializer='Normal')

#optimizer = Optimizer.GradientDescent(model, learning_rate, train_feature, train_label, step_search='Armijo')
optimizer = Optimizer.AcceleratedGradientDescent(model, train_feature, train_label, lr=0)
logger = Functions.Logger("Gradient Descent", start=model.W)
logger.record('TrainLoss', [model.loss(train_feature, train_label)])
logger.record('TestLoss', [model.loss(test_feature, test_label)])
logger.record('TestAccu', [test_label, model.forward(test_feature)])


# 3. Train
print("####  [ Epoch: {:4} / {} ]    [ Accuracy: {:.4f} ]".format(0, epochs, logger.metric_traj['TestAccu'][-1]))
for i in range(epochs):
    model.backward(train_feature, train_label)
    optimizer.step()
    logger.record('TrainLoss', [model.loss(train_feature, train_label)])
    logger.record('TestLoss', [model.loss(test_feature, test_label)])
    logger.record('Gradient', [model.grad])
    logger.record('TestAccu', [test_label, model.forward(test_feature)])
    print("####  [ Epoch: {:4} / {} ]    [ Accuracy: {:.4f} ]".format(i+1, epochs, logger.metric_traj['TestAccu'][-1]))


# 4. Visualize
#logger.plot_metric("TrainLoss")
#logger.plot_metric("TestLoss")
logger.plot_metric("Gradient")
#logger.plot_metric("TestAccu")
