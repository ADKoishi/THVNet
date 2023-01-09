from torch.utils.tensorboard import SummaryWriter


class TrainResult:
    def __init__(self, path: str):
        self.model_path = path
        self.epochs = []
        self.train_losses = []
        self.validation_losses = []
        self.test_loss = 0
        self.save_epochs = []
        self.save_losses = []

    def read_model_result(self):
        with open(self.model_path, 'r') as model_file:
            line = model_file.readline()
            while line:
                line_elements = line.strip('\n').split(', ')
                if len(line_elements) == 5:
                    epoch = int(line_elements[0].split('=')[-1])
                    self.epochs.append(epoch)
                    average_epoch_loss = float(line_elements[2].split('=')[-1])
                    self.train_losses.append(average_epoch_loss)
                if len(line_elements) == 4:
                    average_epoch_loss = float(line_elements[2].split('=')[-1])
                    self.validation_losses.append(average_epoch_loss)
                if len(line_elements) == 1:
                    sub_elements = line_elements[0].split(': ')
                    if len(sub_elements) == 2:
                        self.save_losses.append(float(sub_elements[-1]))
                        self.save_epochs.append(int(sub_elements[0].split(' ')[3]))
                    if len(sub_elements) == 1:
                        self.test_loss = float(sub_elements[-1])

                line = model_file.readline()


writer = SummaryWriter()

# # old_train_losses = []
# # old_validation_losses = []
# # with open("models\M5\M5_OLD\Zero_TransOut16_FC32_ResTrue_BNTrue.ckpt .rst.txt") as f:
# #     line = f.readline()
# #     while line:
# #         if len(line.split(', ')) > 1:
# #             old_train_losses.append(float(line.split(', ')[2].split('=')[-1]))
# #             line = f.readline()
# #             old_validation_losses.append(float(line.split(', ')[2].split('=')[-1]))
# #             line = f.readline()
# #             continue
# #         line = f.readline()
#
# for epoch in range(len(old_train_losses)):
#     writer.add_scalars(
#         'losses',
#         {'fixed_train_losses': old_train_losses[epoch],
#          'fixed_validation_losses': old_validation_losses[epoch]},
#         epoch
#     )

train_result = TrainResult(
    "models/M5/FC8/Zero_TransOut16_TARGET_NUM5_TRANS_OUT_NUM16_TRANS_OUT_DIM120_HIDDEN_DIM120_ACTIVATIONReLU_FC8_ResTrue_BNTrue.ckpt.rst.txt")
train_result.read_model_result()
for epoch in train_result.epochs:
    writer.add_scalars(
        'losses',
        {'120_train': train_result.train_losses[epoch],
         '120_validation': train_result.validation_losses[epoch]},
        epoch
    )
train_result = TrainResult(
    "models/M5/FC8/Zero_TransOut16_TARGET_NUM5_TRANS_OUT_NUM16_TRANS_OUT_DIM240_HIDDEN_DIM240_ACTIVATIONReLU_FC8_ResTrue_BNTrue.ckpt.rst.txt")
train_result.read_model_result()
for epoch in train_result.epochs:
    writer.add_scalars(
        'losses',
        {'240_train': train_result.train_losses[epoch],
         '240_validation': train_result.validation_losses[epoch]},
        epoch
    )
