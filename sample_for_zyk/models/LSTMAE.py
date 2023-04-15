import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import trange
import numpy as np


from .utils.evaluate import evaluate_by_entity
from .utils.log import record_res


class Encoder(nn.Module):
    def __init__(self, num_metric=4096, hidden_dim=1024, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            num_metric,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, hidden_dim)
        outputs, (hidden, cell) = self.lstm(x)
        return (hidden, cell)


class Decoder(nn.Module):
    def __init__(
        self, num_metric=4096, hidden_dim=1024, output_size=4096, num_layers=2
    ):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            num_metric,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x: tensor of shape (batch_size, seq_length, hidden_dim)
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)


class LSTMVAE(nn.Module):
    """LSTM-based Variational Auto Encoder"""

    def __init__(
        self, num_metric, hidden_dim, z_dim
    ):
        """
        num_metric: int, batch_size x sequence_length x num_metric
        hidden_dim: int, output size of LSTM AE
        z_dim: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(LSTMVAE, self).__init__()

        # dimensions
        self.num_metric = num_metric
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_layers = 1

        # lstm ae
        self.lstm_enc = Encoder(
            num_metric=num_metric, hidden_dim=hidden_dim, num_layers=self.num_layers
        )
        self.lstm_dec = Decoder(
            num_metric=z_dim,
            output_size=num_metric,
            hidden_dim=hidden_dim,
            num_layers=self.num_layers,
        )

        self.fc21 = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, self.hidden_dim)

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)

        z = mu + noise * std
        return z

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape

        # encode input space to hidden space
        enc_hidden = self.lstm_enc(x)
        enc_h = enc_hidden[0].view(batch_size, self.hidden_dim)
        # extract latent variable z(hidden space to latent space)
        mean = self.fc21(enc_h)
        logvar = self.fc22(enc_h)
        z = self.reparametize(mean, logvar)  # batch_size x z_dim

        # decode latent space to input space
        z = z.repeat(1, seq_len, 1)
        z = z.view(batch_size, seq_len, self.z_dim)
        reconstruct_output, hidden = self.lstm_dec(z, enc_hidden)

        x_hat = reconstruct_output

        # calculate vae loss
        losses = self.loss_function(x_hat, x, mean, logvar)
        m_loss, recon_loss, kld_loss = (
            losses["loss"],
            losses["Reconstruction_Loss"],
            losses["KLD"],
        )

        return m_loss, x_hat, (recon_loss, kld_loss)

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = 0.00025  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }


class LSTMAE(nn.Module):
    """LSTM-based Auto Encoder"""

    def __init__(self, num_metric, hidden_dim, z_dim):
        """
        num_metric: int, batch_size x sequence_length x input_dim
        hidden_dim: int, output size of LSTM AE
        z_dim: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(LSTMAE, self).__init__()

        # dimensions
        self.num_metric = num_metric
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        # lstm ae
        self.lstm_enc = Encoder(
            num_metric=num_metric,
            hidden_dim=hidden_dim,
        )
        self.lstm_dec = Decoder(
            num_metric=num_metric,
            output_size=num_metric,
            hidden_dim=hidden_dim,
        )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape

        enc_hidden = self.lstm_enc(x)

        temp_input = torch.zeros(
            (batch_size, seq_len, feature_dim), dtype=torch.float)
        hidden = enc_hidden
        reconstruct_output, hidden = self.lstm_dec(
            temp_input, hidden)  # batch_size x seq_len x num_metric

        # 计算编码向量enc_hidden[0]的中心 center 和半径 radius。其中，中心 center 表示编码向量的平均值，半径 radius 表示编码向量到中心的最大距离。
        # 定义损失函数_loss。该函数包括两部分，一部分是重构误差，即输入数据 x 和模型重构的数据 x_recon 的欧式距离平方的均值；另一部分是正则化项，即半径 radius 的平方。
        center = torch.mean(enc_hidden[0], dim=0)
        radius = torch.max(torch.sqrt(
            torch.sum((enc_hidden[0] - center)**2, dim=1)))

        def loss(x, x_recon, center, radius):
            dist = torch.sum((x - x_recon) ** 2, dim=1)
            loss = torch.mean(dist) + radius ** 2
            return loss
        
        reconstruct_loss = loss(x, reconstruct_output, center, radius)
        return reconstruct_loss, reconstruct_output, (0, 0)


def train(dataloader, params, writer, test_loaders=None):
    epoch_cnt = params.get("epoch_cnt")
    num_metric = params.get("num_metric")
    z_dim = params.get("z_dim")
    win_len = params.get("win_len")
    hidden_dim = params.get("hidden_dim")
    lr = params.get("lr")
    device = params.get("device")

    log_interval = 1

    model = LSTMAE(num_metric, hidden_dim, z_dim).to(device)  # type: LSTMVAE
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_ls = []
    global_step = 0
    for epoch in (pbar := trange(epoch_cnt)):
        model.train()
        for step, x in enumerate(dataloader):
            x = x.to(device).view(-1, win_len, num_metric)
            loss, x_recons, (recon_loss, kld_loss) = model(x)
            loss_ls.append(loss.item())

            if (step + 1) % log_interval == 0:
                avg_loss = np.average(loss_ls)
                pbar.set_description(
                    f"Epoch: {epoch+1}, Loss: {avg_loss: .4f}")
                writer.add_scalar("loss", avg_loss, global_step=global_step)
                global_step += 1

                loss_ls.clear()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        if (epoch + 1) % log_interval == 0:
            res_ls = evaluate_by_entity(model, test_loaders, test, params)
            record_res(writer, res_ls, epoch+1)

    return model


def test(model: LSTMAE, dataloader, params):
    win_len = params.get("win_len")
    num_metric = params.get("num_metric")
    device = params.get("device")

    labels, raw_seq, est_seq, loss = [], [], [], []
    model.eval()

    loss_ls = []
    with torch.no_grad():
        for x, y in dataloader:
            labels.append(y.numpy())

            x = x.to(device).view(-1, win_len, num_metric)
            loss, x_recons, (recon_loss, kld_loss) = model(x)

            loss_ls.append(loss.cpu().item())
            raw_seq.append(x[:, -1].cpu().numpy())
            est_seq.append(x_recons[:, -1].cpu().numpy())

    raw_seq = np.concatenate(raw_seq, axis=0)
    est_seq = np.concatenate(est_seq, axis=0)
    labels = np.concatenate(labels, axis=0)

    return raw_seq, est_seq, np.average(loss_ls), labels
