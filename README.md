#### Contrastive Clustering with Robust Continuous Embedding and Reliable Negative Sampling (C-CRCN)

In this work, we propose a novel framework Contrastive Clustering with Robust Continuous Embedding and Reliable Negative Sampling (C-CRCN) for PU learning from heterogeneous graph data with a focus on robust clustering and contrastive learning.The framework is designed to address the challenges of heterogeneous data analysis, especially in the context of positive-unlabeled (PU) learning. 

The key innovations include the integration of graph data enhancement, contrastive pre-training, robust continuous clustering, and reliable negative sample selection. We formalize the approach in a series of steps: 
- contrast pre-training using augmented views of the graph data, 
- joint optimization of contrastive and clustering losses, 
- dynamic clustering based on representative points, 
- reliable negative sample selection, and 
- classifier training and evaluation.

To demonstrate the motivation behind C-CRCN, we constructed a toy dataset that exhibits heterogeneity among positive samples. In this toy dataset, we utilize a one Latent Prototypes method to identify negative samples for classification purposes. The method leverages graph neural networks and contrastive learning to obtain representations of positive samples. The positive prototype is derived by averaging the representations of positive samples. Negative samples are identified by comparing unlabeled samples' similarity to the positive prototype. Subsequently, both positive and negative samples are fed into a logistic regression model to train the classifier.

<img src="https://github.com/MdRTaP/C-CRCN/blob/master/PU%20data%20that%20exhibits%20heterogeneity%20among%20positive%20samples.png" width="300" height="280"><img src="https://github.com/MdRTaP/C-CRCN/blob/master/Identification%20of%20Reliable%20Negative%20Samples%20Based%20on%20the%20positive%20prototype.png" width="300" height="280"/>

Figure 1 illustrates the distribution of PU data and highlights the limitations of relying solely on a single positive prototype to identify reliable negative samples. Specifically, Figure 1(a) shows the distribution of PU data, emphasizing the underlying structure. In Figure 1(b), we demonstrate the attempt to identify trustworthy negative samples based on a single positive prototype. However, all identified samples turn out to be positive instances, not negative ones. This outcome underscores a significant challenge when data exhibits heterogeneity: depending on a single prototype leads to erroneous classification. This flaw highlights the critical shortcomings of such an approach. Through this example, we emphasize the need for our proposed method, which addresses these limitations by accounting for multiple aspects of data heterogeneity, ultimately improving the accuracy of negative sample identification.
