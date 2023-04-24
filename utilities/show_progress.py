import torch
import matplotlib.pyplot as plt

class ShowProgress():
    def __init__(self, typeOfData):
        self.typeOfData = typeOfData

        # loss history
        self.y_loss = {}
        self.y_loss[f'{self.typeOfData}'] = []
        self.y_loss['val'] = []

        self.top5_rate = {}
        self.top5_rate[f'{self.typeOfData}'] = []
        self.top5_rate['val'] = []

        self.top1_rate = {}
        self.top1_rate[f'{self.typeOfData}'] = []
        self.top1_rate['val'] = []

        self.x_epoch = []

        # Create figure
        self.fig = plt.figure()
        self.ax0 = self.fig.add_subplot(131, title="Loss")
        self.ax1 = self.fig.add_subplot(132, title="Top1acc")
        self.ax2 = self.fig.add_subplot(133, title="Top5acc")

    def draw_curve(self, current_epoch):
        self.x_epoch.append(current_epoch)
        self.ax0.plot(self.x_epoch, torch.Tensor(self.y_loss[f'{self.typeOfData}']), 'bo-', label=f'{self.typeOfData}')
        self.ax1.plot(self.x_epoch, torch.Tensor(self.top1_rate[f'{self.typeOfData}']), 'bo-', label=f'{self.typeOfData}')
        self.ax2.plot(self.x_epoch, torch.Tensor(self.top5_rate[f'{self.typeOfData}']), 'bo-', label=f'{self.typeOfData}')
        if current_epoch == 0:
            self.ax0.legend()
            self.ax1.legend()
            self.ax2.legend()
    
    def appendData(self, loss, top1, top5):
        self.y_loss[f'{self.typeOfData}'].append(loss)
        self.top1_rate[f'{self.typeOfData}'].append(top1)
        self.top5_rate[f'{self.typeOfData}'].append(top5)

    def saveFig(self, path):
        self.fig.savefig(path)