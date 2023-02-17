import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_score, confusion_matrix
# standarizer
from sklearn.preprocessing import StandardScaler
import pdb
import json
import os

def prepare_data(pairs_num_per_playlist=1, pair_len=10):
    '''
    This function is to prepare data for training and testing
    Now it is a random data generator and you can change it to your own data by replacing the positive_pairs and negative_pairs
    '''

    print('Preparing data...')
    path = './data_features'
    data = {}
    data['playlists'] = []
    for filename in tqdm(os.listdir(path)):
        if filename.endswith('.json'):
            file_path = os.path.join(path, filename)
            with open(file_path) as f:
                data_slice = json.load(f)
            for i in range(len(data_slice['playlists'])):
                data['playlists'].append(data_slice['playlists'][i])
    # data = json.load(open(filename))
    print('Total number of playlists:', len(data['playlists']))

    playlist = data['playlists'][0]['tracks']
    # print(playlist[0]['features'].keys())
    # print('Total number of tracks in the first playlist:', len(playlist))
    # for i in range(9):
    #     print(data['playlists'][1]['tracks'][i]['track_name'])

    positive_pairs = []
    all_songs_uri = []
    uri_dict={}

    for playlist in tqdm(data['playlists']):
        features = []
        tracks = playlist['tracks']
        uri_list = []

        if len(tracks) < pair_len:
            # print('playlist length not enough')
            continue

        for song in tracks:
            if song.get('features') is None:
                continue
            else:
                song_features = list(song['features'].values())
                song_features = song_features[:11]
                song_features.append(song['track_uri'][14:]) # the first 14 char is same,Discard them for complexity
                features.append(song_features)
                uri_list.append(song['track_uri'][14:])

                # if song['track_uri'] not in uri_dict.keys():

                # all_songs_uri.append(song['track_uri'])  # 400812
            # print(features)

        # generate the dict , uri to features
        for uri in uri_list:
            if uri not in uri_dict.keys():
                uri_dict[uri]={}
                uri_dict[uri]['total_num'] = 1
            uri_dict[uri]['total_num'] =+1
            others_uri_list = [item for item in uri_list if item != uri]
            for uri_other in others_uri_list:
                if uri_other not in uri_dict[uri].keys():
                    uri_dict[uri][uri_other] = 0
                uri_dict[uri][uri_other] =+1


        features = np.array(features)
        # sample 10 tracks from features for per_track_num times

        for i in range(pairs_num_per_playlist):
            indexes = np.random.choice(features.shape[0], pair_len, replace=False)
            positive_pairs.append(features[indexes])

    # all_songs_uri_distinct = list(set(all_songs_uri))  # 121489

    positive_pairs = np.array(positive_pairs)
    print('positive_pairs shape: ', positive_pairs.shape)  # (4630, 10, 13)

    negative_pairs = positive_pairs.copy()
    all_songs_in_pairs = negative_pairs.reshape(-1, negative_pairs.shape[-1])
    # randomly sample shape[0] songs from all_songs_in_pairs
    random_indexes = np.random.choice(all_songs_in_pairs.shape[0], negative_pairs.shape[0], replace=True)
    random_songs = all_songs_in_pairs[random_indexes]
    negative_pairs[:, -1, :] = random_songs
    print('negative_pairs shape: ', negative_pairs.shape)  # (4630, 10, 13)

    # print(negative_pairs==positive_pairs)

    # generate random data
    # positive pairs (total_size, 10, feature_size)
    # negative pairs (total_size, 10, feature_size)
    # labels (total_size, 1)

    total_size = positive_pairs.shape[0]

    # feature_size = 13
    # positive_pairs = np.random.rand(total_size, 10, feature_size) + 0.5  # + 1 to make the positive pairs different from negative pairs
    # negative_pairs = np.random.rand(total_size, 10, feature_size)

    # ones for positive pairs, zeros for negative pairs (another way to generate data)
    # positive_pairs = np.ones((total_size, 10, feature_size))
    # negative_pairs = np.zeros((total_size, 10, feature_size))

    # print(positive_pairs)
    # print(negative_pairs)

    all_pairs = np.concatenate((positive_pairs, negative_pairs), axis=0)

    add_features_all = []
    for i in range(all_pairs.shape[0]):
        add_features = []
        features = all_pairs[i,:,:]
        uri_last = features[-1,-1]
        uri_dict[uri_last]['total_num']
        for j in range(features.shape[0]-1):
            uri_not_last = features[j,-1]
            if uri_not_last in uri_dict[uri_last].keys():
                simple_count = uri_dict[uri_last][uri_not_last]
                Sorenson_index = 2*uri_dict[uri_last][uri_not_last]/(uri_dict[uri_last]['total_num']+uri_dict[uri_not_last]['total_num'])
            else:
                simple_count = 0
                Sorenson_index = 0
            add_features.append(simple_count)
            add_features.append(Sorenson_index)
        add_features_all.append(add_features)
    # len(add_features_all),len(add_features_all[0])  (115800, 18)
    add_features_all = np.array(add_features_all)
    all_pairs = np.delete(all_pairs, -1, axis=2)   # delete the uri



    # standarizing the data
    standarizer = StandardScaler()
    all_pairs = standarizer.fit_transform(all_pairs.reshape(-1, all_pairs.shape[-1])).reshape(all_pairs.shape)
    add_features_all = standarizer.fit_transform(add_features_all)

    # flatten and concat
    all_pairs = all_pairs.reshape(all_pairs.shape[0],-1)
    all_pairs = np.concatenate((all_pairs, add_features_all), axis=1)
    
    # print(all_pairs[:total_size] == all_pairs[total_size:])
    
    # to torch tensor
    # positive_pairs = torch.Tensor(positive_pairs)
    # negative_pairs = torch.Tensor(negative_pairs)

    # concatenate and create labels
    # all_pairs = torch.cat((positive_pairs, negative_pairs), dim=0)
    all_pairs = torch.Tensor(all_pairs)
    all_labels = torch.cat((torch.ones(total_size, 1), torch.zeros(total_size, 1)), dim=0)

    print('all_pairs shape: ', all_pairs.shape)
    print('all_labels shape: ', all_labels.shape)

    # convert to torch dataset

    full_dataset = torch.utils.data.TensorDataset(all_pairs, all_labels)

    return full_dataset


def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, training_threshold, testing_threshold):
    '''
    This function is to train the model
    '''

    print('Training...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    losses = []
    accuracies = []

    test(model, test_loader, criterion, testing_threshold)

    for epoch in range(num_epochs):
        average_loss = 0
        average_acc = 0
        model.train()
        all_pred = []
        all_labels = []
        for data, labels in tqdm(train_loader):
            # move tensors to GPU if CUDA is available
            data = data.to(device)
            labels = labels.to(device)

            # compute output
            outputs = model(data)

            # make prediction based on the threshold
            pred = (outputs.data > training_threshold)
            # convert True/False to 1/0
            pred = pred.float()

            loss = criterion(outputs, labels)
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            average_loss += loss.item()
            all_pred.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_pred = np.concatenate(all_pred)
        all_labels = np.concatenate(all_labels)

        average_acc = np.sum(all_pred == all_labels) / len(all_labels)
        average_precision = precision_score(all_labels, all_pred)
        average_loss = average_loss/len(train_loader)

        print('Epoch [{}/{}], Training Loss: {:.6f}, Training Accuracy: {:.6f}, Training Precision: {:.6f}'.format(epoch+1, num_epochs, average_loss, average_acc, average_precision))

        train_loss = average_loss
        test_loss, test_acc, (all_pred, all_labels) = test(model, test_loader, criterion, testing_threshold)

        losses.append([train_loss, test_loss])
        accuracies.append([average_acc, test_acc])

    losses = np.array(losses)
    accuracies = np.array(accuracies)

    return model, losses, accuracies, (all_pred, all_labels)


def test(model, test_loader, criterion, testing_threshold):
    '''
    This function is to test the model
    '''

    print('Testing...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        average_loss = 0
        all_pred = []
        all_labels = []
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            pred = (outputs.data > testing_threshold)
            pred = pred.float()

            loss = criterion(outputs, labels)
            average_loss += loss.item()
            all_pred.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # pdb.set_trace()
        all_pred = np.concatenate(all_pred)
        all_labels = np.concatenate(all_labels)

        average_acc = np.sum(all_pred == all_labels) / len(all_labels)
        average_precision = precision_score(all_labels, all_pred)
        average_loss = average_loss/len(test_loader)

        print('Test Loss: {:.6f}, Test Accuracy: {:.6f}, Test Precision: {:.6f}'.format(average_loss, average_acc, average_precision))

    return average_loss, average_acc, (all_pred, all_labels)


class RecModel(nn.Module):
    '''
    define a model with:
    four fc layers with ReLU activation, three dropout layers, one sigmoid activation
    node number: 10*feature_size -> 20 -> 7 -> 3 -> 1
    dropout rate: 0.8 -> 0.4 -> 0.2
    '''

    def __init__(self, feature_size):
        super(RecModel, self).__init__()

        self.fc1 = nn.Linear(feature_size, 20)
        self.fc2 = nn.Linear(20, 7)
        self.fc3 = nn.Linear(7, 3)
        self.fc4 = nn.Linear(3, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):

        # print(x.shape)
        # x = x.view(x.shape[0], -1)
        # print(x.shape)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.sigmoid(x)
        
        return x


def main(model, full_dataset, batch_size, learning_rate, num_epochs, training_threshold, testing_threshold):
    '''
    This function is the main function to train and test the model

    full_dataset: torch dataset
    batch_size: int
    learning_rate: float
    num_epochs: int
    '''

    # split into train and test
    # print(len(full_dataset))
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # set num_workers to 4 or 8 if you have a GPU
    if torch.cuda.is_available():
        num_workers = 4
    else:
        num_workers = 0

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # a binary classification task with sigmoid activation
    criterion = nn.BCELoss()
    model, losses, accuracies, (all_pred, all_labels) = train(model, train_loader, test_loader, criterion, optimizer, num_epochs, training_threshold, testing_threshold)

    # visualize the loss as the network trained

    # show the loss
    fig = plt.figure(figsize=(10, 8))
    train_losses = losses[:, 0]
    test_losses = losses[:, 1]
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses')
    plt.legend()
    plt.savefig('losses.png')

    # show the accuracy
    fig = plt.figure(figsize=(10, 8))
    train_acc = accuracies[:, 0]
    test_acc = accuracies[:, 1]
    plt.plot(range(1, num_epochs+1), train_acc, label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), test_acc, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracies')
    plt.legend()
    plt.savefig('accuracies.png')

    # analyze the results
    confusion = confusion_matrix(all_labels, all_pred)
    precision = precision_score(all_labels, all_pred)
    print('Confusion Matrix:\n', confusion)
    print('Precision: ', precision)


if __name__ == '__main__':

    #filename = 'mpd.slice.115000-115999_withfeatures.json'

    full_dataset = prepare_data()

    # hyperparameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 20
    training_threshold = 0.5
    testing_threshold = 0.5

    # initialize the model
    feature_size = full_dataset[0][0].shape[0]
    model = RecModel(feature_size=feature_size)
    
    # train and test the model
    main(model, full_dataset, batch_size, learning_rate, num_epochs, training_threshold, testing_threshold)
