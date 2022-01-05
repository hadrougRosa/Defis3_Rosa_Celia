######Celia Boulahouat & Rosa Hadroug######
########### MASTER 2 SICOM ###############
import torch
import torch.nn as nn
import string
import random
import sys
import unidecode
import time
import math
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import  SummaryWriter
# Configuration de la gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# recuperer tout les caracteres
all_characters = string.printable
n_characters = len(all_characters)

# lecture de notre corpus
file = unidecode.unidecode(open("Partie_Apprentissage/CorpusFiltredFinal.txt").read())


class LSTM(nn.Module):
    #définir les couches du modèle
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #une couche d'intégration pour rechercher les vecteurs d'intégration pour la séquence d'entrée.
        self.embed = nn.Embedding(input_size, hidden_size)
        #un module lstm standard pour calculer l'activation de l'état caché et les fonctionnalités de sortie
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        #une couche linéaire pour appliquer une transformation linéaire aux entités de sortie du module lstm.
        self.fc = nn.Linear(hidden_size, output_size)

    # cette methode prendra une séquence d'entrée et les états précédents 
    # et produira la sortie avec les états du pas de temps actuel 
    def forward(self, x, hidden, cell):
        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, (hidden, cell)
    #Parce que nous devons réinitialiser les états au début de chaque époque, 
    #donc on a définit une méthode supplémentaire pour nous aider à remettre tous les états à zéro
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell
#Calcul le temps de trainnig(debut a la fin)
def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

class Generator:
    def __init__(self):
        self.chunk_len = 250
        self.num_epochs = 1000
        self.batch_size = 1
        self.print_every = 10
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.003
        
    

    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_characters.index(string[c])
        return tensor

    def get_random_batch(self):
        start_idx = random.randint(0, len(file) - self.chunk_len)
        end_idx = start_idx + self.chunk_len + 1
        text_str = file[start_idx:end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            text_input[i, :] = self.char_tensor(text_str[:-1])
            text_target[i, :] = self.char_tensor(text_str[1:])

        return text_input.long(), text_target.long()

    def generate(self, initial_str="a", predict_len=100, temperature=0.85):
        hidden, cell = self.lstm.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.lstm(
                initial_input[p].view(1).to(device), hidden, cell
            )

        last_char = initial_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.lstm(
                last_char.view(1).to(device), hidden, cell
            )
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_char]
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)

        return predicted
    
    
    # input_size, hidden_size, num_layers, output_size
    def train(self):
        #Déclarer le modèle LSTM
        self.lstm = LSTM(
            n_characters, self.hidden_size, self.num_layers, n_characters
        ).to(device)
        #Nous utilisons une valeur de taux d'apprentissage de 0.003
        #ET une fonction d'optimisation adam
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        #writer = SummaryWriter(f"N/runs/passwords")  # for tensorboard

        print("=> Starting training")
        total_loss =0
        all_losses= []
        n_iters = 100000
        plot_every = 400
        start = time.time()
        for epoch in range(1, self.num_epochs + 1):
            inp, target = self.get_random_batch()
            hidden, cell = self.lstm.init_hidden(batch_size=self.batch_size)

            self.lstm.zero_grad()
            loss = 0
            #mettre les donnee soit dans cuda ou cpu
            inp = inp.to(device)
            target = target.to(device)

            for c in range(self.chunk_len):
                output, (hidden, cell) = self.lstm(inp[:, c], hidden, cell)
                loss += criterion(output, target[:, c])
                total_loss +=loss
            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len
            File2 = open("Partie_Apprentissage/PredirePass.txt", "a+")
            if epoch % self.print_every == 0:
                #print(f"Loss: {loss}")
                print('%s (%d %.3f) %.4f' % (timeSince(start), epoch,epoch/n_iters*100 , loss))
                line = self.generate()
                File2.write(line)
                File2.write("\r")
                File2.close() 
            #writer.add_scalar("Training loss", loss, global_step=epoch)
        
            if epoch % plot_every == 0:
                with torch.no_grad():
                    all_losses.append(loss)
                    loss = 0
        plt.figure()
        plt.ylabel("loss")
        plt.xlabel("epoch*10")
        plt.title('trainingLoss')
        plt.plot(all_losses)
        #plt.show()
genpasswords = Generator()
genpasswords.train()