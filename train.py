import torch
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from dataser import *
from model import *
from userloss import *
from torch.optim.lr_scheduler import StepLR
from zlceval import *
from unit import *
from torch.utils.tensorboard import SummaryWriter
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
writer = SummaryWriter()

setup_seed(3407)

loss = MSELoss()
md = KpDataset('./dataset/train/',shuffle=True,lenght=1000)
evalDataset = KpDataset('./dataset/test/')
net = FullConModel()
net.cuda()
input =torch.randn(1, 1, 256, 256).cuda()
writer.add_graph(net, input)
dl = DataLoader(md,batch_size=16)
accum_step=1
num_epochs = 1000

# Adagrad Adam SparseAdam AdamW ASGD LBFGS RMSprop Rprop
# Adadelta
optimizer = torch.optim.SGD( net.parameters() , lr=0.01)
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

bset_ecval = 999
epoch_last_loss = 0
step =0
for epoch in range(1, num_epochs + 1):
    pbar = enumerate(dl)
    for i, (imgs, targets,_,_) in iter(pbar):
        step+=1
        imgs=imgs.cuda()
        targets=targets.cuda()
        net_out=net(imgs)
        l = loss(net_out, targets)
        epoch_last_loss = l
        l = l/accum_step
        l.backward()
        if step % accum_step ==0:
            optimizer.step()
            optimizer.zero_grad()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)  # clip gradient
    scheduler.step()
    # ecval = zlceval(net,evalDataset,loss)
    if l<bset_ecval:
        bset_ecval=l
        torch.save(net,"net_best.pt")
    logging.info('Epoch %d, Loss: %f', epoch, epoch_last_loss)
    writer.add_scalar('Loss/train', epoch_last_loss, epoch)
writer.close()
torch.save(net,"net.pt")
