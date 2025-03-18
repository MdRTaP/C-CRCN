from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, pairwise_distances
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from augment import augment_graph, agmt_dict
from blp import *
from load_dataset_and_preprocess import *
import argparse
import torch.nn as nn
from scipy.sparse import csr_matrix, triu
from scipy.sparse.csgraph import connected_components
import warnings
from net import GCNEncoder, MLP_Predictor
warnings.filterwarnings("ignore")


def train_pretrained_representations():
    for epoch in range(1, 1000):
        model.train()
        optimizer.zero_grad()
        g1, g2 = augment_1(data), augment_2(data)
        p1, aux_h2 = model(g1, g2)
        p2, aux_h1 = model(g2, g1)

        unlabeled_loss = predict_unlabeled_nodes(p1, aux_h2.detach(), p2, aux_h1.detach())
        pretrained_loss = unlabeled_loss

        pretrained_loss.backward()
        optimizer.step()
        model.update_aux_network(0.005)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, training loss：{pretrained_loss.item():.4f}')
def rcc_loss(X: torch.Tensor, U: torch.Tensor, edge_index: torch.Tensor, lambda_: float = 0.5,
                gamma: float = 3.0) -> torch.Tensor:
    data_term = 0.5 * torch.sum((X - U) ** 2)

    if edge_index.size(1) == 0:
        return data_term

    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    weights = torch.exp(-torch.norm(X[src_nodes] - X[dst_nodes], dim=1))

    edge_distances = torch.norm(U[src_nodes] - U[dst_nodes], dim=1)

    abs_dist = torch.abs(edge_distances)
    mcp_values = torch.where(abs_dist <= gamma * lambda_, lambda_ * abs_dist - 0.5 * (edge_distances ** 2) / gamma,
                                0.5 * gamma * lambda_ ** 2)

    connection_term = 0.5 * lambda_ * torch.sum(weights * mcp_values)

    return data_term + connection_term


class RCCModel(nn.Module):
    def __init__(self, X):
        super().__init__()
        self.device = X.device
        self.U = nn.Parameter(X.detach().clone())
        self.edge_index = self._build_mutual_knn(X.detach().cpu())

    def _build_mutual_knn(self, X, k=5):
        X_np = X.cpu().detach().numpy()
        samples = X_np.shape[0]
        batch_size = 10000

        indices = np.zeros((samples, k), dtype=np.int32)

        for x in range(0, samples, batch_size):
            start = x
            end = min(x + batch_size, samples)

            dist_batch = np.linalg.norm(
                X_np[start:end, None] - X_np[None, :],
                axis=-1
            )

            partitioned = np.argpartition(dist_batch, kth=k, axis=1)
            indices[start:end] = partitioned[:, 1:k + 1]

        row = np.repeat(np.arange(samples), k)
        col = indices.ravel()
        adj_matrix = csr_matrix(
            (np.ones_like(row), (row, col)),
            shape=(samples, samples)
        )

        mutual_adj = adj_matrix.minimum(adj_matrix.transpose())

        upper_tri = triu(mutual_adj, k=1)
        edge_pairs = np.asarray(upper_tri.nonzero()).T

        return torch.from_numpy(edge_pairs).t().contiguous().long()

    def get_representatives(self):
        return self.U.detach()

def get_cluster_labels(U, delta=10):
    U_np = U.cpu().numpy()
    dist_matrix = np.sqrt(((U_np[:, np.newaxis] - U_np) ** 2).sum(axis=2))

    adj_matrix = (dist_matrix < delta).astype(int)

    n_components, labels = connected_components(
        csr_matrix(adj_matrix),
        directed=False,
        connection='weak'
    )

    cluster_centers = np.array([U_np[labels == i].mean(axis=0)
                                for i in range(n_components)])

    silhouette_avg = silhouette_score(U_np, labels)

    return labels, silhouette_avg, cluster_centers

def train_rcc(model, X, loss_fn, epochs=500, lr=0.1, delta=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(X, model.U, model.edge_index)
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_history.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}")

    U_final = model.U.detach().cpu()
    labels, silhouette_avg, cluster_centers = get_cluster_labels(U_final, delta)

    return labels, silhouette_avg, cluster_centers, loss_history


def prepare_data(positive_nodes , reliable_negative_nodes,embeddings):
    positive_features = embeddings[positive_nodes].detach().cpu().numpy()
    positive_labels= np.ones(len(positive_nodes))

    negative_features=embeddings[reliable_negative_nodes].detach().cpu().numpy()
    negative_labels=np.zeros(len(reliable_negative_nodes))

    X = np.vstack((positive_features,negative_features))
    y = np.hstack((positive_labels, negative_labels))
    return X, y

def train_network(epochs):
    model.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        g1, g2 = augment_1(data), augment_2(data)
        p1, aux_h2 = model(g1, g2)
        p2, aux_h1 = model(g2, g1)

        RCC_p1 = RCCModel(p1[positive_nodes].detach())
        positive_clustering_loss1 = rcc_loss(p1[positive_nodes], RCC_p1.U, RCC_p1.edge_index)
        RCC_p2 = RCCModel(p2[positive_nodes].detach())
        positive_clustering_loss2 = rcc_loss(p2[positive_nodes], RCC_p2.U, RCC_p2.edge_index)

        unlabeled_loss = predict_unlabeled_nodes(p1, aux_h2.detach(), p2, aux_h1.detach())
        positive_loss = (positive_clustering_loss1 + positive_clustering_loss2) / 2

        # joint learning
        if unlabeled_loss > positive_loss:
            loss = unlabeled_loss
        else:
            loss = positive_loss

        loss.backward()
        optimizer.step()
        model.update_aux_network(0.005)  # update auxiliary network
        if epoch % 100 == 0:
            print(f'Joint learning loss Epoch{epoch}/{epochs},Loss:{loss.item():.4f}')

    torch.save(model.state_dict(), 'trained_model.pth')
    print('Model training is complete and weights have been saved to trained_model.pth')
def train_classifier(X_train , y_train):
    classifier = LogisticRegression()
    classifier.fit(X_train , y_train)
    return classifier

def evaluate_classifier(classifier,X_test,y_test):
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    f1 = f1_score(y_test , y_pred)

    return accuracy , f1

def train_and_evaluate_classifier(positive_nodes,reliable_negative_nodes,embeddings):
    X, y = prepare_data(positive_nodes , reliable_negative_nodes , embeddings)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    classifier = train_classifier(X_train,y_train)

    accuracy,f1 = evaluate_classifier(classifier, X_test,y_test)
    print(f'Classifier accuracy：{accuracy:.4f}')
    print(f'Classifier F1 score：{f1:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='load parameter for running and evaluating \
                                                 boostrap pu learning')
    parser.add_argument('--dataset', '-d', type=str, default='amazon-photos',
                        help='Data set to be used')
    parser.add_argument('--positive_index', '-c', type=int, default=0,
                        help='Index of label to be used as positive')
    parser.add_argument('--sample_seed', '-s', type=int, default=1,
                        help='random seed for sample labeled positive from all positive nodes')
    parser.add_argument('--train_pct', '-p', type=float, default=0.2,
                        help='Percentage of positive nodes to be used as training positive')
    parser.add_argument('--val_pct', '-v', type=float, default=0.1,
                        help='Percentage of positive nodes to be used as evaluating positive')
    parser.add_argument('--test_pct', '-t', type=float, default=1.00,
                        help='Percentage of unknown nodes to be used as test set')
    parser.add_argument('--hidden_size', '-l', type=int, default=32,
                        help='Size of hidden layers')
    parser.add_argument('--output_size', '-o', type=int, default=16,
                        help='Dimension of output representations')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load pu dataset
    data = load_dataset(args.dataset)
    data = make_pu_dataset(data, pos_index=[args.positive_index], sample_seed=args.sample_seed,
                           train_pct=args.train_pct, val_pct=args.val_pct,test_pct=args.test_pct)
    data = data.to(device)
    dataset = [data]

    # Prepare augment
    drop_edge_p_1, drop_feat_p_1, drop_edge_p_2, drop_feat_p_2 = agmt_dict[args.dataset]
    augment_1 = augment_graph(drop_edge_p_1, drop_feat_p_1)
    augment_2 = augment_graph(drop_edge_p_2, drop_feat_p_2)

    # Build BLP networks
    input_size = data.x.size(1)
    encoder = GCNEncoder(input_size, args.hidden_size, args.output_size)
    predictor = MLP_Predictor(args.output_size, args.hidden_size, args.output_size)
    model = BLP(encoder, predictor).to(device)

    optimizer = AdamW(model.trainable_parameters(), lr=5e-4, weight_decay=1e-5)
    positive_nodes = data.train_mask.nonzero(as_tuple=False).view(-1)   #74 positive labelled in Amazon photos.
    unlabeled_nodes = (~data.train_mask).nonzero(as_tuple=False).view(-1)
    val_nodes = data.val_mask.nonzero(as_tuple=False).view(-1)


    # Contrast Pre-Training
    train_pretrained_representations()

    model.eval()
    with torch.no_grad():
        g1,g2=augment_1(data),augment_2(data)
        p1_pretrained_final,_  = model(g1,g2)
        p2_pretrained_final,_  = model(g2,g1)

    p1_pretrained_final_labeled_positive=p1_pretrained_final[positive_nodes].detach().cpu().numpy()
    p2_pretrained_final_labeled_positive=p2_pretrained_final[positive_nodes].detach().cpu().numpy()

    torch.save(model.state_dict(),'pretrained_model.pth')

    tsne=TSNE(n_components=2)
    p1_pretrained_final_labeled_positive_2d= tsne.fit_transform(p1_pretrained_final_labeled_positive)

    plt.figure(figsize=(8,6))
    plt.scatter(p1_pretrained_final_labeled_positive_2d[:,0],p1_pretrained_final_labeled_positive_2d[:,1],label='Nodes',color='blue',alpha=0.7)
    plt.title=('positive in p1 obtained after pre-training')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Joint Optimization Phase
    train_network(epochs=5000)

    g1,g2=augment_1(data),augment_2(data)
    p1,_ = model(g1,g2)
    p2,_ = model(g2,g1)

    # Dynamic Clustering and Positive Sample Segmentation
    p1_trained_final_labeled_positive = p1[positive_nodes]
    p2_trained_final_labeled_positive = p2[positive_nodes]
    RCC_p1_trained = RCCModel(p1_trained_final_labeled_positive)
    RCC_p2_trained = RCCModel(p2_trained_final_labeled_positive)
    labels_p1, silhouette_avg_p1,p1_cluster_centers, losses = train_rcc(RCC_p1_trained, p1_trained_final_labeled_positive, rcc_loss, delta=10)
    labels_p2, silhouette_avg_p2,p2_cluster_centers, losses = train_rcc(RCC_p2_trained, p2_trained_final_labeled_positive, rcc_loss, delta=10)
    p1_n_clusters = len(set(labels_p1))
    p2_n_clusters = len(set(labels_p2))


    # Reliable Negative Sample Selection
    unlabeled_embeddings=p1[unlabeled_nodes]
    p1_cluster_centers_tensors = torch.from_numpy(p1_cluster_centers).to(unlabeled_embeddings.device)
    min_similarities1 = torch.zeros(len(unlabeled_embeddings)).to(unlabeled_embeddings.device)

    for i,node_embedding in enumerate(unlabeled_embeddings):
        similarities=cosine_similarity(node_embedding.unsqueeze(0),p1_cluster_centers_tensors)
        min_similarities1[i]=torch.min(similarities)

    unlabeled_embeddings = p2[unlabeled_nodes]
    p2_cluster_centers_tensors = torch.from_numpy(p2_cluster_centers).to(unlabeled_embeddings.device)
    min_similarities2 = torch.zeros(len(unlabeled_embeddings)).to(unlabeled_embeddings.device)

    for i, node_embedding in enumerate(unlabeled_embeddings):
        similarities = cosine_similarity(node_embedding.unsqueeze(0), p2_cluster_centers_tensors)
        min_similarities2[i] = torch.min(similarities)

    final_similarities = min_similarities1 * min_similarities2

    find_reliable_negative_nums=len(positive_nodes)

    bottom_num_indices= torch.topk(-final_similarities,k=find_reliable_negative_nums).indices
    bottom_num_similarities= final_similarities[bottom_num_indices]
    reliable_negative_nodes = unlabeled_nodes[bottom_num_indices]

    train_and_evaluate_classifier(positive_nodes, reliable_negative_nodes,p1)
    train_and_evaluate_classifier(positive_nodes, reliable_negative_nodes, p2)
