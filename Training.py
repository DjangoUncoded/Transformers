from Transformer import *
from Masking_Batching import batches
class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1,
               target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        output = self.criterion(x, true_dist.detach())
        return output

    # %%
class NoamOpt:
        def __init__(self, model_size, factor, warmup, optimizer):
            self.optimizer = optimizer
            self._step = 0
            self.warmup = warmup
            self.factor = factor
            self.model_size = model_size
            self._rate = 0

        def step(self):
            self._step += 1
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p['lr'] = rate
            self._rate = rate
            self.optimizer.step()

        def rate(self, step=None):
            if step is None:
                step = self._step
            output = self.factor * (self.model_size ** (-0.5) *
                                    min(step ** (-0.5), step * self.warmup ** (-1.5)))
            return output


#%%
class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()


optimizer = NoamOpt(256, 1, 2000, torch.optim.Adam(
    model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
criterion = LabelSmoothing(trg_vocab,
                           padding_idx=0, smoothing=0.0)
loss_func = SimpleLossCompute(
            model.generator, criterion, optimizer)

#%%
for epoch in range(50):
    model.train()
    tloss=0
    tokens=0
    for batch in batches:
        out = model(batch.src, batch.trg,
                    batch.src_mask, batch.trg_mask)
        loss = loss_func(out, batch.trg_y, batch.ntokens)
        tloss += loss
        tokens += batch.ntokens
    print(f"Epoch {epoch}, average loss: {tloss/tokens}")
torch.save(model.state_dict(),"files/de2en.pth")


#you can Ignore the training and simply just Copy the below step in your notebook
model.load_state_dict(torch.load("files/de2en.pth"))

