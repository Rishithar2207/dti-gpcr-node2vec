

# 1. Install and Import Required Packages
# Originally a Colab cell magic. Dependencies now live in requirements.txt:
#   pip install -r requirements.txt

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap.umap_ as umap
import random
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 2. Load and Preprocess the Datasets
dti_df = pd.read_csv("gpcr_admat_dgc.txt", sep='\t', index_col=0)
drug_sim_df = pd.read_csv("gpcr_simmat_dc.txt", sep='\t', index_col=0)
target_sim_df = pd.read_csv("gpcr_simmat_dg.txt", sep='\t', index_col=0)

# Convert all IDs to lowercase for consistency
dti_df.columns = dti_df.columns.str.lower()
dti_df.index = dti_df.index.str.lower()
drug_sim_df.columns = drug_sim_df.columns.str.lower()
drug_sim_df.index = drug_sim_df.index.str.lower()
target_sim_df.columns = target_sim_df.columns.str.lower()
target_sim_df.index = target_sim_df.index.str.lower()

drug_list = drug_sim_df.index.tolist()
target_list = target_sim_df.index.tolist()
dti_matrix = dti_df.loc[target_list, drug_list].values

print(f"DTI matrix shape: {dti_matrix.shape}")
print(f"Drug similarity matrix shape: {drug_sim_df.shape}")
print(f"Target similarity matrix shape: {target_sim_df.shape}")

# 3. Construct the Graph
def add_top_k_edges(G, matrix, nodes, k, edge_type):
    for i, node_i in enumerate(nodes):
        scores = matrix[i].copy()
        scores[i] = -np.inf  # avoid self-connections
        top_k_idx = np.argpartition(-scores, k)[:k]
        for j in top_k_idx:
            node_j = nodes[j]
            if not G.has_edge(node_i, node_j):
                G.add_edge(node_i, node_j, weight=matrix[i][j], edge_type=edge_type)
    return G

def build_graph(drugs, targets, dti_mat, drug_sim, target_sim, top_k=5):
    G = nx.Graph()
    G.add_nodes_from(drugs, bipartite='drugs', node_type='drug')
    G.add_nodes_from(targets, bipartite='targets', node_type='target')
    for i, target in enumerate(targets):
        for j, drug in enumerate(drugs):
            if dti_mat[i][j] == 1:
                G.add_edge(drug, target, weight=1.0, edge_type='dti')
    G = add_top_k_edges(G, drug_sim.values, drugs, top_k, 'drug_sim')
    G = add_top_k_edges(G, target_sim.values, targets, top_k, 'target_sim')
    return G

G = build_graph(drug_list, target_list, dti_matrix, drug_sim_df, target_sim_df, top_k=5)
print(f"Final Graph: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")

# 4. Define Embedding Functions
def get_node2vec_embeddings(graph, dimensions=128, walk_length=80, num_walks=10, p=1.0, q=0.5, workers=4, seed=42):
    node2vec = Node2Vec(
        graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks,
        p=p, q=q, workers=workers, weight_key='weight', seed=seed
    )
    model = node2vec.fit(window=5, min_count=1)
    embeddings = {node: model.wv[node] for node in graph.nodes()}
    return embeddings

def get_deepwalk_embeddings(graph, dimensions=128, walk_length=80, num_walks=10, workers=4, seed=42):
    return get_node2vec_embeddings(graph, dimensions, walk_length, num_walks, p=1.0, q=1.0, workers=workers, seed=seed)

# 5. Perform Stratified Negative Sampling
def stratified_negative_sampling(positive_edges, drugs, targets):
    all_pairs = set((d, t) for d in drugs for t in targets)
    negative_pool = list(all_pairs - positive_edges)
    negative_samples = []

    drug_to_targets_pos = {d: set() for d in drugs}
    target_to_drugs_pos = {t: set() for t in targets}
    for d, t in positive_edges:
        drug_to_targets_pos[d].add(t)
        target_to_drugs_pos[t].add(d)

    for d, t in positive_edges:
        neg_t_candidates = set(targets) - drug_to_targets_pos[d]
        if neg_t_candidates:
            neg_t = random.choice(list(neg_t_candidates))
            negative_samples.append((d, neg_t, 0))

        neg_d_candidates = set(drugs) - target_to_drugs_pos[t]
        if neg_d_candidates:
            neg_d = random.choice(list(neg_d_candidates))
            negative_samples.append((neg_d, t, 0))

    needed = len(positive_edges) - len(negative_samples)
    if needed > 0:
        additional_negatives = random.sample(negative_pool, needed)
        negative_samples.extend([(d, t, 0) for d, t in additional_negatives])

    return negative_samples[:len(positive_edges)]

# 6. Generate Positive and Negative Samples
positive_edges = set((d, t) for i, t in enumerate(target_list) for j, d in enumerate(drug_list) if dti_matrix[i][j] == 1)
positive_samples = [(d, t, 1) for d, t in positive_edges]
negative_samples = stratified_negative_sampling(positive_edges, drug_list, target_list)
samples = positive_samples + negative_samples
random.shuffle(samples)

# 7. Generate Node2Vec Embeddings
embeddings = get_node2vec_embeddings(G)

# 8. Build the Feature Matrix and Label Vector
X, y = [], []
for d, t, label in samples:
    if d in embeddings and t in embeddings:
        X.append(np.concatenate([embeddings[d], embeddings[t]]))
        y.append(label)
X = np.array(X)
y = np.array(y)

# 9. Tune Hyperparameters for Classifiers
param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100]}
param_grid_xgb = {'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n Logistic Regression Hyperparameter Tuning")
lr = LogisticRegression(max_iter=1000)
grid_lr = GridSearchCV(lr, param_grid_lr, scoring='roc_auc', cv=cv, verbose=1, n_jobs=-1)
grid_lr.fit(X_scaled, y)
print(f"Best LR params: {grid_lr.best_params_}, Best ROC-AUC: {grid_lr.best_score_:.4f}")

print("\n XGBoost Hyperparameter Tuning ")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
grid_xgb = GridSearchCV(xgb, param_grid_xgb, scoring='roc_auc', cv=cv, verbose=1, n_jobs=-1)
grid_xgb.fit(X_scaled, y)
print(f"Best XGB params: {grid_xgb.best_params_}, Best ROC-AUC: {grid_xgb.best_score_:.4f}")

best_lr = grid_lr.best_estimator_
best_xgb = grid_xgb.best_estimator_

# 10. Cross-validation Performance Evaluation
def cross_val_eval(model, X, y, cv):
    roc_aucs = []
    pr_aucs = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        model.fit(X[train_idx], y[train_idx])
        y_prob = model.predict_proba(X[test_idx])[:, 1]
        roc_aucs.append(roc_auc_score(y[test_idx], y_prob))
        pr_aucs.append(average_precision_score(y[test_idx], y_prob))
        print(f"Fold {fold} - ROC-AUC: {roc_aucs[-1]:.4f}, PR-AUC: {pr_aucs[-1]:.4f}")
    return roc_aucs, pr_aucs

print("\n Cross-validation Logistic Regression ")
roc_lrs, pr_lrs = cross_val_eval(best_lr, X_scaled, y, cv)
print(f"LR ROC-AUC Mean ± 95% CI: {np.mean(roc_lrs):.4f} ± {1.96*np.std(roc_lrs)/np.sqrt(len(roc_lrs)):.4f}")
print(f"LR PR-AUC  Mean ± 95% CI: {np.mean(pr_lrs):.4f} ± {1.96*np.std(pr_lrs)/np.sqrt(len(pr_lrs)):.4f}")

print("\n Cross-validation XGBoost ")
roc_xgbs, pr_xgbs = cross_val_eval(best_xgb, X_scaled, y, cv)
print(f"XGB ROC-AUC Mean ± 95% CI: {np.mean(roc_xgbs):.4f} ± {1.96*np.std(roc_xgbs)/np.sqrt(len(roc_xgbs)):.4f}")
print(f"XGB PR-AUC  Mean ± 95% CI: {np.mean(pr_xgbs):.4f} ± {1.96*np.std(pr_xgbs)/np.sqrt(len(pr_xgbs)):.4f}")

# 11. Baseline Comparison using DeepWalk
embeddings_dw = get_deepwalk_embeddings(G)
X_dw = [np.concatenate([embeddings_dw[d], embeddings_dw[t]]) for d, t, _ in samples if d in embeddings_dw and t in embeddings_dw]
X_dw = np.array(X_dw)
X_dw_scaled = scaler.fit_transform(X_dw)

print("\n Baseline DeepWalk + Logistic Regression ")
grid_lr.fit(X_dw_scaled, y)
print(f"DeepWalk LR Best params: {grid_lr.best_params_}, Best ROC-AUC: {grid_lr.best_score_:.4f}")

# 12. Additional Classifier: Random Forest
from sklearn.ensemble import RandomForestClassifier

param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
rf = RandomForestClassifier(random_state=42)

print("\n Random Forest Hyperparameter Tuning ")
grid_rf = GridSearchCV(rf, param_grid_rf, scoring='roc_auc', cv=cv, verbose=1, n_jobs=-1)
grid_rf.fit(X_scaled, y)
print(f"Best RF params: {grid_rf.best_params_}, Best ROC-AUC: {grid_rf.best_score_:.4f}")

best_rf = grid_rf.best_estimator_

# 13. Final Evaluation on Hold-out Test Set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

for name, model in [('Logistic Regression', best_lr), ('XGBoost', best_xgb), ('Random Forest', best_rf)]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} Final Test Results:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC : {average_precision_score(y_test, y_prob):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# 14. Error Analysis - Top False Positives & Negatives
def error_analysis(model, X_test, y_test, samples_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    fp_idx = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred)) if yt == 0 and yp == 1]
    fn_idx = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred)) if yt == 1 and yp == 0]

    print(f"\nNumber of False Positives: {len(fp_idx)}")
    print(f"Number of False Negatives: {len(fn_idx)}")

    print("\nSample False Positives (Predicted interaction but no known interaction):")
    for i in fp_idx[:5]:
        d, t, _ = samples_test[i]
        print(f"Drug: {d}, Target: {t}, Score: {y_prob[i]:.4f}")

    print("\nSample False Negatives (Missed known interactions):")
    for i in fn_idx[:5]:
        d, t, _ = samples_test[i]
        print(f"Drug: {d}, Target: {t}, Score: {y_prob[i]:.4f}")

# Prepare test samples specifically for error analysis
_, samples_test = train_test_split(samples, stratify=y, test_size=0.2, random_state=42)

print("\n Error Analysis for Logistic Regression ")
error_analysis(best_lr, X_test, y_test, samples_test)

# 15. Visualizations with Legends
X_emb = np.array([embeddings[n] for n in embeddings])
labels = ['drug' if n in drug_list else 'target' for n in embeddings]

X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_emb)
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, alpha=0.7)
plt.title("t-SNE Node2Vec Embeddings")
plt.legend(title="Node Type", loc='upper right', bbox_to_anchor=(1.15, 1))
plt.show()

X_umap = umap.UMAP(random_state=42).fit_transform(X_emb)
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels, alpha=0.7)
plt.title("UMAP Node2Vec Embeddings")
plt.legend(title="Node Type", loc='upper right', bbox_to_anchor=(1.15, 1))
plt.show()

# 16. Save Outputs
embedding_df = pd.DataFrame(embeddings).T
embedding_df.to_csv("node_embeddings.csv")
print("Saved node embeddings to 'node_embeddings.csv'")

pred_df = pd.DataFrame(samples, columns=['drug', 'target', 'label'])
pred_df.to_csv("dti_samples.csv", index=False)
print("Saved samples to 'dti_samples.csv'")

nx.write_gexf(G, "dti_graph.gexf")
print("Saved graph to 'dti_graph.gexf'")

# 17. Ethical Note
print("\nEthical Note: All predictions are computational in nature and require biological validation before clinical or experimental use.")

from sklearn.decomposition import TruncatedSVD

print("\nMatrix Factorization (SVD) Embeddings Baseline")

# Perform SVD on the DTI matrix to obtain a low-dimensional representation
# DTI matrix: rows = targets, columns = drugs
n_components = min(128, dti_matrix.shape[0], dti_matrix.shape[1])
svd = TruncatedSVD(n_components=n_components, random_state=42)

# Generate embeddings separately for drugs and targets
svd_drug = svd.fit_transform(dti_matrix.T)
svd_target = svd.fit_transform(dti_matrix)

# Combine embeddings into a single dictionary
svd_embeddings = {}
for i, d in enumerate(drug_list):
    svd_embeddings[d] = svd_drug[i]
for i, t in enumerate(target_list):
    svd_embeddings[t] = svd_target[i]

# Construct the input feature matrix by concatenating drug and target embeddings
X_svd = []
for d, t, label in samples:
    if d in svd_embeddings and t in svd_embeddings:
        X_svd.append(np.concatenate([svd_embeddings[d], svd_embeddings[t]]))
X_svd = np.array(X_svd)
X_svd_scaled = scaler.fit_transform(X_svd)

# Train a logistic regression classifier using the SVD-based features
lr_svd = LogisticRegression(max_iter=1000)
roc_svd = []
pr_svd = []
for train_idx, test_idx in cv.split(X_svd_scaled, y):
    lr_svd.fit(X_svd_scaled[train_idx], y[train_idx])
    probs = lr_svd.predict_proba(X_svd_scaled[test_idx])[:, 1]
    roc_svd.append(roc_auc_score(y[test_idx], probs))
    pr_svd.append(average_precision_score(y[test_idx], probs))

print(f"SVD ROC-AUC: {np.mean(roc_svd):.4f} ± {np.std(roc_svd):.4f}")
print(f"SVD PR-AUC : {np.mean(pr_svd):.4f} ± {np.std(pr_svd):.4f}")
print("\nHard Negative Mining")

# First-round embeddings are assumed to be available (Node2Vec pre-trained)
# Generate all possible drug-target pairs and identify the known positives
all_pairs = [(d, t) for d in drug_list for t in target_list]
positive_set = set((d, t) for d, t, l in samples if l == 1)
negative_candidates = [pair for pair in all_pairs if pair not in positive_set]

# Compute cosine similarity scores for all candidate negatives
def get_score(d, t):
    if d in embeddings and t in embeddings:
        return cosine_similarity([embeddings[d]], [embeddings[t]])[0][0]
    return -1

scored_negatives = [(d, t, get_score(d, t)) for d, t in negative_candidates]
scored_negatives.sort(key=lambda x: x[2], reverse=True)

# Select the most challenging negatives (those most similar to positives)
hard_negatives = [(d, t, 0) for d, t, s in scored_negatives[:len(positive_samples)]]

# Create a balanced dataset of positives and hard negatives
samples_hard = positive_samples + hard_negatives
random.shuffle(samples_hard)

X_hard = []
y_hard = []
for d, t, label in samples_hard:
    if d in embeddings and t in embeddings:
        X_hard.append(np.concatenate([embeddings[d], embeddings[t]]))
        y_hard.append(label)

X_hard = np.array(X_hard)
y_hard = np.array(y_hard)
X_hard_scaled = scaler.fit_transform(X_hard)

# Evaluate logistic regression classifier on the hard negative set
lr_hard = LogisticRegression(max_iter=1000)
roc_hard = []
pr_hard = []
for train_idx, test_idx in cv.split(X_hard_scaled, y_hard):
    lr_hard.fit(X_hard_scaled[train_idx], y_hard[train_idx])
    probs = lr_hard.predict_proba(X_hard_scaled[test_idx])[:, 1]
    roc_hard.append(roc_auc_score(y_hard[test_idx], probs))
    pr_hard.append(average_precision_score(y_hard[test_idx], probs))

print(f"Hard Negatives ROC-AUC: {np.mean(roc_hard):.4f} ± {np.std(roc_hard):.4f}")
print(f"Hard Negatives PR-AUC : {np.mean(pr_hard):.4f} ± {np.std(pr_hard):.4f}")
print("\nTop-K Novel Predictions (Unknown Interactions)")

from operator import itemgetter

# Score all unknown drug–target pairs using cosine similarity
predictions = []
for d in drug_list:
    for t in target_list:
        if (d, t) not in positive_set:
            score = cosine_similarity([embeddings[d]], [embeddings[t]])[0][0]
            predictions.append((d, t, score))

# Sort predictions by similarity and show top-10 novel interactions
predictions.sort(key=itemgetter(2), reverse=True)
top_k_novel = predictions[:10]

print("Top 10 Novel DTI Predictions:")
for i, (d, t, score) in enumerate(top_k_novel, 1):
    print(f"{i}. Drug: {d}, Target: {t}, Score: {score:.4f}")
# These can be looked up in external databases (e.g., ChEMBL, UniProt) for biological validation

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier

print("\nMLP + Stacking Classifier Evaluation")

# Define an MLP classifier for deep feature learning
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)

# Combine classifiers using stacking with logistic regression as the meta-learner
stack_model = StackingClassifier(
    estimators=[('lr', best_lr), ('xgb', best_xgb), ('mlp', mlp)],
    final_estimator=LogisticRegression()
)

# Evaluate stacked model using cross-validation
roc_stack = []
pr_stack = []
for train_idx, test_idx in cv.split(X_scaled, y):
    stack_model.fit(X_scaled[train_idx], y[train_idx])
    probs = stack_model.predict_proba(X_scaled[test_idx])[:, 1]
    roc_stack.append(roc_auc_score(y[test_idx], probs))
    pr_stack.append(average_precision_score(y[test_idx], probs))

print(f"Stacked ROC-AUC: {np.mean(roc_stack):.4f} ± {np.std(roc_stack):.4f}")
print(f"Stacked PR-AUC : {np.mean(pr_stack):.4f} ± {np.std(pr_stack):.4f}")
from scipy.stats import ttest_rel

print("\nPaired Significance Test (ROC-AUC)")

# Perform a paired t-test to compare logistic regression vs stacking performance
t_stat, p_value = ttest_rel(roc_lrs, roc_stack)
print(f"Paired t-test between LR and Stacking ROC-AUCs: t={t_stat:.3f}, p={p_value:.4f}")
if p_value < 0.05:
    print(" Statistically significant improvement.")
else:
    print(" Not statistically significant.")
print("\nFeature Importances")

# Visualize feature importances from XGBoost and Random Forest
xgb_importances = best_xgb.feature_importances_
rf_importances = best_rf.feature_importances_

plt.figure(figsize=(10, 4))
plt.bar(range(len(xgb_importances)), xgb_importances)
plt.title("XGBoost Feature Importances")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()

plt.figure(figsize=(10, 4))
plt.bar(range(len(rf_importances)), rf_importances)
plt.title("Random Forest Feature Importances")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()

print("\nExternal Validation Hook")

# Optional: additional benchmark datasets (e.g., NR, IC) for generalizability testing
print("This is to optionally validate generalization using an external benchmark dataset (e.g., Yamanishi NR or IC)")


# Example usage:
# external_dti_df = pd.read_csv("nr_admat_dgc.txt", sep='\t', index_col=0)
# external_drug_sim_df = pd.read_csv("nr_simmat_dc.txt", sep='\t', index_col=0)
# external_target_sim_df = pd.read_csv("nr_simmat_dg.txt", sep='\t', index_col=0)
# [Run full pipeline again using external data]

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from node2vec import Node2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import random
import warnings

warnings.filterwarnings("ignore")

# 18. Node2Vec Hyperparameter Tuning
def tune_node2vec(graph, samples, cv, scaler, param_grid=None):
    """
    Perform grid search to tune Node2Vec hyperparameters and evaluate downstream
    classification performance using Logistic Regression.
    """
    if param_grid is None:
        param_grid = {
            'p': [0.5, 1.0, 2.0],
            'q': [0.5, 1.0, 2.0],
            'dimensions': [64, 128, 256],
            'walk_length': [40, 80],
            'num_walks': [10, 20]
        }

    best_score = 0
    best_params = {}

    for params in ParameterGrid(param_grid):
        print(f"Testing Node2Vec params: {params}")
        embeddings = get_node2vec_embeddings(
            graph,
            dimensions=params['dimensions'],
            walk_length=params['walk_length'],
            num_walks=params['num_walks'],
            p=params['p'],
            q=params['q'],
            workers=4,
            seed=42
        )

        X = []
        y = []
        for d, t, label in samples:
            if d in embeddings and t in embeddings:
                X.append(np.concatenate([embeddings[d], embeddings[t]]))
                y.append(label)

        X = scaler.fit_transform(np.array(X))
        y = np.array(y)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(lr, X, y, cv=cv, scoring='roc_auc')
        mean_score = np.mean(scores)
        print(f"Mean ROC-AUC for params: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    print(f"\nBest Node2Vec params: {best_params}")
    print(f"Best ROC-AUC: {best_score:.4f}")
    return best_params, best_score

# 19. External Dataset Validation
def evaluate_external_dataset(dataset_path, drug_sim_path, target_sim_path, top_k=5):
    """
    Perform evaluation of the pipeline using an external dataset like Yamanishi NR/IC.
    This function reuses earlier components to load, preprocess, embed, and evaluate.
    """
    print("\n External Dataset Validation ")

    try:
        external_dti_df = pd.read_csv(dataset_path, sep='\t', index_col=0)
        external_drug_sim_df = pd.read_csv(drug_sim_path, sep='\t', index_col=0)
        external_target_sim_df = pd.read_csv(target_sim_path, sep='\t', index_col=0)
    except FileNotFoundError:
        print(" External dataset files not found. Please provide valid paths.")
        return None

    external_dti_df.columns = external_dti_df.columns.str.lower()
    external_dti_df.index = external_dti_df.index.str.lower()
    external_drug_sim_df.columns = external_drug_sim_df.columns.str.lower()
    external_drug_sim_df.index = external_drug_sim_df.index.str.lower()
    external_target_sim_df.columns = external_target_sim_df.columns.str.lower()
    external_target_sim_df.index = external_target_sim_df.index.str.lower()

    external_drug_list = external_drug_sim_df.index.tolist()
    external_target_list = external_target_sim_df.index.tolist()
    external_dti_matrix = external_dti_df.loc[external_target_list, external_drug_list].values

    G_external = build_graph(
        external_drug_list,
        external_target_list,
        external_dti_matrix,
        external_drug_sim_df,
        external_target_sim_df,
        top_k=top_k
    )

    print(f"External Graph: Nodes={G_external.number_of_nodes()}, Edges={G_external.number_of_edges()}")

    embeddings_external = get_node2vec_embeddings(G_external)

    positive_edges = set((d, t) for i, t in enumerate(external_target_list)
                         for j, d in enumerate(external_drug_list) if external_dti_matrix[i][j] == 1)
    positive_samples = [(d, t, 1) for d, t in positive_edges]
    negative_samples = stratified_negative_sampling(positive_edges, external_drug_list, external_target_list)
    samples_external = positive_samples + negative_samples
    random.shuffle(samples_external)

    X_external = []
    y_external = []
    for d, t, label in samples_external:
        if d in embeddings_external and t in embeddings_external:
            X_external.append(np.concatenate([embeddings_external[d], embeddings_external[t]]))
            y_external.append(label)

    X_external = scaler.fit_transform(np.array(X_external))
    y_external = np.array(y_external)

    best_lr.fit(X_external, y_external)
    y_prob = best_lr.predict_proba(X_external)[:, 1]

    metrics = {
        'roc_auc': roc_auc_score(y_external, y_prob),
        'pr_auc': average_precision_score(y_external, y_prob),
        'f1': f1_score(y_external, best_lr.predict(X_external)),
        'mcc': matthews_corrcoef(y_external, best_lr.predict(X_external))
    }

    print("External Dataset Metrics:")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")

    return metrics

# 20. Graph Visualization
def visualize_graph(G, title="DTI Graph Visualization"):
    """
    Visualize the DTI graph using a spring layout. Drugs and targets are displayed
    in different colours for clarity.
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    node_colors = ['blue' if G.nodes[n]['node_type'] == 'drug' else 'red' for n in G.nodes]

    nx.draw(G, pos, node_color=node_colors, node_size=50, alpha=0.7, with_labels=False)
    plt.title(title)
    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Drug'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Target')
        ],
        title="Node Type",
        loc='upper right',
        bbox_to_anchor=(1.15, 1)
    )
    plt.show()

# 21. Extended Evaluation Metrics
def extended_evaluation(model, X_test, y_test, model_name="Model"):
    """
    Generate a more complete set of evaluation metrics for a trained classifier,
    including ROC-AUC, PR-AUC, F1-score, and MCC.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': average_precision_score(y_test, y_prob),
        'f1': f1_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }

    print(f"\n{model_name} Extended Test Results:")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")

    return metrics


# Perform tuning to find best Node2Vec parameters
print("\n Node2Vec Hyperparameter Tuning ")
best_n2v_params, best_n2v_score = tune_node2vec(G, samples, cv, scaler)

# Generate Node2Vec embeddings using the best parameters obtained from tuning
optimal_embeddings = get_node2vec_embeddings(
    G,
    dimensions=best_n2v_params['dimensions'],
    walk_length=best_n2v_params['walk_length'],
    num_walks=best_n2v_params['num_walks'],
    p=best_n2v_params['p'],
    q=best_n2v_params['q'],
    workers=4,
    seed=42
)

# Construct the final feature matrix using the optimal embeddings
X_optimal = []
y_optimal = []
for d, t, label in samples:
    if d in optimal_embeddings and t in optimal_embeddings:
        X_optimal.append(np.concatenate([optimal_embeddings[d], optimal_embeddings[t]]))
        y_optimal.append(label)

X_optimal = np.array(X_optimal)
y_optimal = np.array(y_optimal)
X_optimal_scaled = scaler.fit_transform(X_optimal)

# External dataset validation (paths are placeholders, should be replaced with actual files)
external_metrics = evaluate_external_dataset(
    "nr_admat_dgc.txt",
    "nr_simmat_dc.txt",
    "nr_simmat_dg.txt"
)

# Visualize the graph structure
print("\nGraph Visualization")
visualize_graph(G, title="GPCR DTI Graph Visualization")

# Perform extended evaluation across all classifiers
print("\nExtended Evaluation for All Models")
for name, model in [('Logistic Regression', best_lr), ('XGBoost', best_xgb), ('Random Forest', best_rf)]:
    model.fit(X_train, y_train)
    extended_evaluation(model, X_test, y_test, model_name=name)

# Save the optimal embeddings to file
optimal_embedding_df = pd.DataFrame(optimal_embeddings).T
optimal_embedding_df.to_csv("optimal_node_embeddings.csv")
print(" Saved optimal node embeddings to 'optimal_node_embeddings.csv'")



# Visual Analytics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap.umap_ as umap

# Plot the degree distribution for nodes, separated by type
degrees = dict(G.degree())
degree_values = list(degrees.values())
node_types = [G.nodes[n]['node_type'] for n in G.nodes]

plt.figure(figsize=(10,6))
sns.histplot(data=pd.DataFrame({'Degree': degree_values, 'NodeType': node_types}),
             x='Degree', hue='NodeType', multiple='stack', bins=30)
plt.title("Graph Degree Distribution by Node Type")
plt.xlabel("Node Degree")
plt.ylabel("Count")
plt.show()

# Heatmap of the adjacency matrix (top 100 nodes for visibility)
top_100_nodes = list(G.nodes)[:100]
adj_matrix = nx.to_numpy_array(G, nodelist=top_100_nodes, weight='weight')

plt.figure(figsize=(12,10))
sns.heatmap(adj_matrix, cmap='viridis', xticklabels=False, yticklabels=False)
plt.title("Adjacency Matrix Heatmap (Top 100 Nodes)")
plt.show()

# Compute and plot cosine similarity heatmap for top-K drug-target pairs
# 29. Additional Visualizations
# Graph degree distribution
degrees = dict(G.degree())
degree_values = list(degrees.values())
node_types = [G.nodes[n]['node_type'] for n in G.nodes]

plt.figure(figsize=(10, 6))
sns.histplot(data=pd.DataFrame({'Degree': degree_values, 'NodeType': node_types}),
             x='Degree', hue='NodeType', multiple='stack', bins=30)
plt.title("Graph Degree Distribution by Node Type")
plt.xlabel("Node Degree")
plt.ylabel("Count")
plt.show()

# Adjacency matrix heatmap
top_100_nodes = list(G.nodes)[:100]
adj_matrix = nx.to_numpy_array(G, nodelist=top_100_nodes, weight='weight')

plt.figure(figsize=(12, 10))
sns.heatmap(adj_matrix, cmap='viridis', xticklabels=False, yticklabels=False)
plt.title("Adjacency Matrix Heatmap (Top 100 Nodes)")
plt.show()

# New Drug-target cosine similarity heatmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

top_k = 100  # You can adjust this value

# 1. Get valid embeddings
drug_embeddings = {d: embeddings[d] for d in drug_list if d in embeddings}
target_embeddings = {t: embeddings[t] for t in target_list if t in embeddings}

# 2. Build full cosine similarity matrix
cosine_mat = np.zeros((len(drug_list), len(target_list)))
for i, d in enumerate(drug_list):
    for j, t in enumerate(target_list):
        if d in embeddings and t in embeddings:
            cosine_mat[i, j] = cosine_similarity([embeddings[d]], [embeddings[t]])[0][0]
        else:
            cosine_mat[i, j] = 0

# 3. Extract Top-K highest scoring pairs
flat_indices = np.argsort(cosine_mat, axis=None)[::-1][:top_k]
rows, cols = np.unravel_index(flat_indices, cosine_mat.shape)

# 4. Get unique top-K drug and target indices
unique_drug_indices = np.unique(rows)
unique_target_indices = np.unique(cols)

# 5. Subset cosine matrix to only top-K drugs/targets
topk_matrix = cosine_mat[np.ix_(unique_drug_indices, unique_target_indices)]

# 6. Get labels
topk_drugs = [drug_list[i] for i in unique_drug_indices]
topk_targets = [target_list[j] for j in unique_target_indices]

# 7. Plot clean heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(topk_matrix.T, cmap='coolwarm', xticklabels=topk_drugs, yticklabels=topk_targets)
plt.title(f"Top-{top_k} Drug–Target Cosine Similarity Heatmap")
plt.xlabel("Drugs")
plt.ylabel("Targets")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



# Generate per-target histograms of cosine similarity scores
cosine_scores_long = []
for j, t in enumerate(target_list):
    for i, d in enumerate(drug_list):
        if d in embeddings and t in embeddings:
            cosine_scores_long.append({'Drug': d, 'Target': t, 'Score': cosine_mat[i, j]})

cosine_df = pd.DataFrame(cosine_scores_long)

g = sns.FacetGrid(cosine_df, col="Target", col_wrap=5, height=2, sharex=True, sharey=True)
g.map_dataframe(sns.histplot, x="Score", bins=15)
g.set_titles(col_template="{col_name}")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Cosine Score Distributions per Target")
plt.show()

# Distribution plot of top-K cosine similarity scores
top_k_scores = cosine_mat.flatten()[flat_indices]

plt.figure(figsize=(8,5))
sns.histplot(top_k_scores, bins=20, kde=True, color='purple')
plt.title(f"Cosine Similarity Distribution of Top-{top_k} Predictions")
plt.xlabel("Cosine Similarity Score")
plt.ylabel("Frequency")
plt.show()

# Frequency barplot showing which targets appear most frequently in top predictions
top_targets = [target_list[j] for j in cols]
target_counts = pd.Series(top_targets).value_counts()

plt.figure(figsize=(12,5))
sns.barplot(x=target_counts.index, y=target_counts.values, palette="viridis")
plt.title(f"Frequency of Targets in Top-{top_k} Predicted Novel DTIs")
plt.xlabel("Target")
plt.ylabel("Count in Top Predictions")
plt.xticks(rotation=90)
plt.show()

# Violin plot showing score distributions for most frequent targets
top_targets_set = target_counts.index[:10]
subset_df = cosine_df[cosine_df['Target'].isin(top_targets_set)]

plt.figure(figsize=(12,6))
sns.violinplot(x='Target', y='Score', data=subset_df, inner="quartile", palette="muted")
plt.title("Cosine Score Distributions for Top 10 Frequent Targets in Top-K Predictions")
plt.xticks(rotation=90)
plt.show()

# t-SNE and UMAP projections of the learned node embeddings
X_emb = np.array([embeddings[n] for n in embeddings])
labels = ['Drug' if n in drug_list else 'Target' for n in embeddings]

# t-SNE visualization
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_emb)
plt.figure(figsize=(10,7))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=labels, alpha=0.7, palette=['blue','red'])
plt.title("t-SNE Visualization of Node2Vec Embeddings")
plt.legend(title="Node Type", loc='best')
plt.show()

# UMAP visualization
X_umap = umap.UMAP(random_state=42).fit_transform(X_emb)
plt.figure(figsize=(10,7))
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=labels, alpha=0.7, palette=['blue','red'])
plt.title("UMAP Visualization of Node2Vec Embeddings")
plt.legend(title="Node Type", loc='best')
plt.show()

# Display top-5 drug–target predictions with known/novel label
all_predictions = []
for d in drug_list:
    for t in target_list:
        score = cosine_similarity([embeddings[d]], [embeddings[t]])[0][0] if d in embeddings and t in embeddings else 0
        label_known = 1 if (d, t) in positive_edges else 0
        all_predictions.append({'Drug': d, 'Target': t, 'Score': score, 'Known/Novel': 'Known' if label_known == 1 else 'Novel'})

pred_df_full = pd.DataFrame(all_predictions).sort_values('Score', ascending=False)
print("\nTop 5 Drug-Target Predictions (Known/Novel):")
print(pred_df_full.head(5))

# Ethical disclaimer to close the pipeline
print("\nEthical Disclaimer: These predictions are computational and should be validated biologically before any clinical or experimental application.")

# Optional ablation: graph built without similarity-based edges
def build_graph_no_sim(drugs, targets, dti_mat):
    G_no_sim = nx.Graph()
    G_no_sim.add_nodes_from(drugs, bipartite='drugs', node_type='drug')
    G_no_sim.add_nodes_from(targets, bipartite='targets', node_type='target')
    for i, target in enumerate(targets):
        for j, drug in enumerate(drugs):
            if dti_mat[i][j] == 1:
                G_no_sim.add_edge(drug, target, weight=1.0, edge_type='dti')
    return G_no_sim

G_no_sim = build_graph_no_sim(drug_list, target_list, dti_matrix)
print(f"\nAblation Graph without similarity edges: Nodes={G_no_sim.number_of_nodes()}, Edges={G_no_sim.number_of_edges()}")
