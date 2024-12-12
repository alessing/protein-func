# Protein Function Prediction using Multi-Task Graph Learning

## Motivation

Proteins are macromolecules present in all living things that carry out their core functions. Characterizing the structure and function of proteins has, thus, been a focus of biochemistry, with attempts at solving the problem of protein structure prediction, stretching back decades[1]. Recently, the Nobel Prize in chemistry was awarded for AlphaFold, which made massive advances in ML-based protein structure prediction[2]. However, prediction of protein function, such as  RNA binding and metal ion binding, remains an open challenge with substantial implications for biochemistry and medicine[3].

Our goal in this project was to predict, from the structure of the protein, which functions held-out proteins will and will not enable or contribute to.

The current state-of-the-art performance in protein function prediction is achieved by PhiGNet [4], which does not utilize structural predictions such as those from AlphaFold to create input graphs. Instead, it opts for forming graphs based on the Residue Communities (RCs) and Evolutionary Couples (ECs) found in protein sequences, and uses ESM-1b \cite{Rives622803} for the embeddings of residues. PhiGNet achieved an average AUPR score of 0.80 and an Fmax value of 0.81 when predicting molecular functions from a dataset comprised of GO (Gene Ontology) function annotations. The authors criticized structural-based predictions, as structures predicted by neural networks are not always accurate. To compensate for this, we weighted the protein examples in our training data by a confidence score. We calculated this confidence score by averaging the confidence score of each residue in the predicted AlphaFold v2 [5] structures. 

We represented these protein structures as graphs. Each atom was a node, and the distances between atoms and the atom types defined the edges.

## Protein Function Data

We used the most recent Homo Sapien Gene Association File compiled by the GO Consortium as a dataset for associating protein functions to specific proteins[6]. Not all the functions associated with each human protein are known, but since individual proteins only have a small subset of all possible functions, we assume that a protein does not have a function if it is not known as having that function. There are about 1,000 protein functions in the dataset that we will classify, and there are approximately 20,000 known human proteins. As seen in figures below, the number of functions in proteins exhibits a power law, whereby only a small number of proteins have a large number of functions.

<table>
<tr>
    <td><img src="blog/power_law.png" width="450" alt="First image"></td>
    <td><img src="blog/power_law_scaled.png" width="450" alt="Second image"></td>
</tr>
<tr>
    <td>Histogram of Number of Protein Functions</td>
    <td>Zoomed-in Histogram.</td>
</tr>
</table>

We first applied a filter to the Gene Association File, so that only associations regarding proteins and whether or not they enable/contribute to different functions remained. We also filtered out duplicate rows that showed the same relationship between a protein and a function with different evidence. We then applied a "groupby" operation over protein UniProtKB IDs to get a dataset of all known functions of each known human protein.

We then matched each protein UniProtKB ID in that dataset with its corresponding predicted AlphaFold structure. Using the structure, we defined the graph of each protein. We used a contact map to form edges between nodes, with atoms less than 3Å apart having an edge between them. For node features, we concatenated atom type with cycles of up to 10, 20, or 30. How many cycles were used for node features was a tunable hyperparameter.

We created a 80-10-10 train-val-test split in our dataset of 18,483 proteins. The model was trained on all available data for proteins within the training set (i.e. structure and all available protein function labels). For held-out proteins in the val/test sets, the goal was to correctly predict the functions each would enable, contribute to, or not enable/contribute to.

Since our dataset primarily contained "enables" labels (rather than contributes to/not enables) and most proteins do not enable nor contribute to most protein functions, we augmented each training batch with randomly chosen functions that we labeled as "not enabled" by the associated protein. We added an equal number of negative samples for each protein in a batch as there were positive samples. We created negative samples the same way when evaluating on the validation set.

## Appropriateness & explanation of model(s) (10 points)

In the work of Kanatsoulis et. al. \cite{charilaos}, it was shown if two graphs have adjacency matrices with different eigenvalues, then there is a Graph Neural Network (GNN) that can always distinguish graph isomorphism. That is, a Graph Isomorphism Network (GIN) \cite{gin} with initial structural node features is \textit{strictly} more powerful than the Weisfeiler-Lehman (WL) isomorphism test. Moreover, almost all real graphs satisfy the condition of having different eigenvalues \cite{charilaos}.

Leveraging the intuition of \cite{charilaos}, we augment the node features with the diagonals of the adjacency matrix, which makes the GNN explicitly more powerful than the WL test. In this setting, we are not given an explicit adjacency matrix. Thus, we must infer an adjacency matrix vis-à-vis chemistry simulation software, like PSI4, an edge inference network, as per \cite{egnn}, or by using a contact map \cite{deepfri}. For the moment, in our code, we simply infer the presence of bonds based on the distances between pairs of atoms and the types of each atoms in the pair. However, we plan to build upon this by the end of the project.

![model_architecture](/blog/FuncGNN.png)

In the naive setting, the feature for node $i$ will be its atom type, a number representing the number of protons in the atom. To make the GNN more expressive, the feature for node $i$ will be augmented by:
\begin{equation}
    \textbf{s}_i=[\text{diag}(A^0)_{i,i}, \text{diag}(A^1)_{i,i} \dots, \text{diag}(A^{K-1})_{i,i}]\in\R^K
\end{equation} where $\text{diag}(A^k)_{i,i}$ is the number of $k$-length cycles node $i$ resides in. For the $N$ atoms in the predicted protein structure, we use this structural data, atom type, and coordinates, to make a $3$-way classification prediction whether the protein (1) \textbf{contributes to}, (2) \textbf{enables}, or (3) \textbf{does not enable/contribute to} a specific function.

More precisely, we combine the atom type and structural features to form the initial features for node $i$:
\begin{equation}
    \textbf{h}^{(0)}_i = [\textbf{s}_i, z_i]\in\R^{K+1}
\end{equation} where $z_i$ is the atom type of node $i$. Then, we obtain an embedding representation for each node using a graph neural network as follows:
\begin{equation}
    \textbf{P}:=f_{\theta}(H^{(0)}, X^{(0)})
\end{equation} where $H^{(0)}\in\R^{N\times (K+1)}$ is the matrix of features and $X^{(0)}\in\R^{N\times 3}$ is the matrix of coordinates for the protein.

Before making a classification for each task (protein function), we embed each protein using a GNN. Specifically, for $f_{\theta}$, we experiment with an $E(3)$-invariant graph neural network, as well as a standard GAT-based GNN.

\subsubsection{$E(n)$-Equivariant Graph Neural Networks}

The $E(3)$-invariant network for $f_{\theta}$ preserves symmetries with respect to the coordinates $\x$. That is, the functional properties of a protein are invariant to rotational and translational symmetries. More precisely, for node $i$: 
\begin{equation}
    f_{\theta}(\textbf{h}_i, Q\x_i + \bb) = f_{\theta}(\textbf{h}_i, \x_i )
\end{equation} for $Q\in\R^{3\times 3}$ an orthogonal rotation matrix and $\bb\in\R^3$ a translation vector.

In particular, we stack a sequence of $E(n)$-Equivariant Graph Neural Networks (EGNNs) \cite{egnn}. The $E(3)$-equivariant network at layer $l$ is defined as \cite{egnn}:
\begin{equation}
    \begin{split}
        \m_{ij} &= \phi_e\left(\textbf{h}_i^{(l)}, \h_j^{(l)}, ||\x_i^{(l)}-\x_j^{(l)}||_2^2, a_{ij}\right)\\
        \x_i^{(l+1)}&=\x_i^{(l)}+C\sum_{j\ne i} (\x_i^{(l)} - \x_j^{(l)})\phi_x(\m_{ij})\\
        \m_i&=\sum_{j\ne i}\m_{ij}\\
        \h_i^{(l+1)}&=\phi_h(\h_i^{(l)}, \m_i)
    \end{split}
\end{equation}
for $\phi_e:\R^{2D + 2}\to \R^h$, $\phi_x:\R^h\to\R$, and $\phi_h:\R^{D+h}\to \R^D$ MLPs, and $a_{ij}=(A)_{ij}$ an entry in the adjacency matrix of the protein graph.

\subsubsection{Graph Attention Networks}

The Graph Attention Network (GAT) \cite{gat} employs self-attention to obtain latent representations. In particular, for input node features $\h_i\in\R^F$, for each node $i$ in a graph, we first apply a shared linear transformation $W\in\R^{F'\times F}$. Then a self-attention mechanism $a:\R^{F'}\times \R^{F'}\to\R$ is applied to nodes:
\begin{equation}
    e_{ij}=a(W\h_i,W\h_j),
\end{equation}
which gives the importance of node $j$'s features to node $i$'s features. We obtain normalized weights by using a softmax to normalize across nodes in its neighborhood $\mathcal{N}_i$ \cite{gat}:
\begin{equation}
    \alpha_{ij}=\text{softmax}_j(e_{ij})=\frac{\exp(e_{ij})}{\sum_{k\in\mathcal{N}_i}\exp(e_{ik})}
\end{equation} where $a$ is a single-layer MLP with LeakyReLU activation. Lastly, the final output features are a linear combination of neighboring node features, with an applied nonlinearity $\sigma$:
\begin{equation}
    \h_i'=\sigma\left(\sum_{j\in\mathcal{N}_i} W\h_j\right).
\end{equation}

\subsubsection{Pooling Layer}

The embeddings of the nodes in the protein graph are aggregated to obtain a single representation for the protein, using a pooling layer.

\subsection{EGNN $E(3)$-Invariant Pooling Layer}
For the EGNN, we define an $E(3)$-invariant pooling layer, where we perform an average over $N$ aggregated messages, to get a protein embedding:
\begin{equation}
    \begin{split}
        \m_{ij} &= \phi_e\left(\textbf{h}_i^{(l)}, \h_j^{(l)}, ||\x_i^{(l)}-\x_j^{(l)}||_2^2, a_{ij}\right)\\
        \m_i&=\sum_{j\ne i}\m_{ij}\\
        \p&=\frac{1}{N}\sum_{i=1}^N \m_i\in\R^D
    \end{split}
\end{equation}

This is $E(3)$-invariant because $\phi_e$ is $E(3)$-invariant to coordinate transformations, as shown by \cite{egnn}.

\subsection{GAT Pooling Layer}

The GAT layer simply applies global mean pooling:
\begin{equation}
    \p=\frac{1}{N}\sum_{i=1}^N \h_i'\in\R^D
\end{equation} where $\h_i'$ is the final-layer embedding for atom $i$.

% at layer $l$ is defined as:
% \begin{equation}
%     \begin{split}
%         \m_{ij}&=\Psi_e(\h_i^{(l)}, \h_j^{(l)}, ||\x_i^{(l)}-\x_j^{(l)}||_2)\\
%         \m_i &= \sum_{j\ne i}\m_{ij}\\
%         \textbf{p} &= \frac{1}{N}\sum_{i=1}^N \m_i
%     \end{split}
% \end{equation}
% Finally, we apply a linear layer and compute a probability, between $0$ and $1$:
% \begin{equation}
%     \hat{y} = \sigma(W\p + \bb).
% \end{equation} where $W\in\R^{1\times D}$ and $\sigma:\R\to[0,1]$ is a sigmoid activation.

\subsection{Multi-task Learning for Protein Functions}

During test time, for each protein $b$, the objective is to predict if a protein possesses functions associated with it, i.e. the set of functions $\mathcal{T}^{(b)}$. This is a multi-task problem, whereby we hold out proteins during training and, at test time, we perform $|\mathcal{T}^{(b)}|$ 3-way classifications. Specifically, for each protein $b$ and atom $i$, we process input coordinates $\x_i\in\R^3$ and features $\h_i\in\R^{D}$ with a neural network $f_{\theta}:\R^{D}\times \R^3\to \R^D$ to obtain a hidden protein representation $\p^{(b)}\in\R^D$. 

Furthermore, for each protein, we perform a $3$-way classification, conditioned on each task corresponding to that protein. Thus, for a batch of $B$ proteins and a set of tasks $\mathcal{T}_b$ for protein $b$, we compute the cross entropy loss as follows:
\begin{equation}
    \sum_{b=1}^{B} \sum_{t^{(b)}=j_1}^{j_{|\mathcal{T}_b|}} \mathcal{H}\left(y^{(t^{(b)})}, g_{\phi}(\textbf{t}^{(b)}, \textbf{p}^{(b)})\right)
\end{equation}

where $\mathbf{t}^{(b)}=\text{Embed}(t^{(b)})\in\R^{E}$ is the task embedding corresponding to task index $t^{(i)}$, $\mathbf{p}^{(b)}=f_{\theta}(X^{(b)}, H^{(b)})$ is the embedding for protein $b$ with coordinates $X^{(b)}\in\R^{N\times 3}$ and features $H^{(b)}\in\R^{N\times (K+1)}$, and $g_{\phi}:\R^{E}\times \R^{D}\to[0,1]^3$ is an MLP with a softmax applied to its logits. That is, $g_{\phi}(\textbf{t}^{(b)}, \textbf{p}^{(b)})\in[0,1]^3$ gives the discrete probability distribution that the protein $b$ contributes to, enables, or does not enable/contribute to protein function $t^{(b)}$.

## Insights + results (10 points)

The primary metric used is an F1 score. The F1 score was computed with respect to three classes: (1) contributes, (2) enables, or (3) does not contribute to/enable a function. In particular, we treated the \textit{not enables} class as the negative class, while both \textit{contributes to} and \textit{enables} are treated as the positive class. The metric is computed according to true positives (TP), false positives (FP) and false negatives (FN) as follows:
\begin{equation}
    F_1=\frac{TP}{TP + \frac{1}{2}(FP + FN)}.
\end{equation}

We used early-stopping to prevent model over-fitting. More precisely, we early-stopped once the cross entropy loss started to increase on our validation set.

## Figures (10 points)
## Code snippets (10 points)

## References
[1] \citep{1973_protein_folding}

[2] \citep{alphafold}

[3] \citep{jang-2024}

[4] \cite{phignet}

[5] \cite{alphafold}

[6] \cite{goa}