arxiv = """
everything is Connected: Graph Neural Networks
Petar Veliˇckovi´c
DeepMind / University of Cambridge
In many ways, graphs are the main modality of data we receive from na-
ture. This is due to the fact that most of the patterns we see, both in natural
and artificial systems, are elegantly representable using the language of graph
structures. Prominent examples include molecules (represented as graphs of
atoms and bonds), social networks and transportation networks. This poten-
tial has already been seen by key scientific and industrial groups, with already-
impacted application areas including traffic forecasting, drug discovery, social
network analysis and recommender systems. Further, some of the most success-
ful domains of application for machine learning in previous years—images, text
and speech processing—can be seen as special cases of graph representation
learning, and consequently there has been significant exchange of information
between these areas. The main aim of this short survey is to enable the reader
to assimilate the key concepts in the area, and position graph representation
learning in a proper context with related fields.
1
arXiv:2301.08210v1 [cs.LG] 19 Jan 2023
1 Introduction: Why study data on graphs?
In this survey, I will present a vibrant and exciting area of deep learning research:
graph representation learning. Or, put simply, building machine learning models over
data that lives on graphs (interconnected structures of nodes connected by edges).
These models are commonly known as graph neural networks, or GNNs for short.
There is very good reason to study data on graphs. From the molecule (a graph of
atoms connected by chemical bonds) all the way to the connectomic structure of the
brain (a graph of neurons connected by synapses), graphs are a universal language
for describing living organisms, at all levels of organisation. Similarly, most relevant
artificial constructs of interest to humans, from the transportation network (a graph
of intersections connected by roads) to the social network (a graph of users connected
by friendship links), are best reasoned about in terms of graphs.
This potential has been realised in recent years by both scientific and indus-
trial groups, with GNNs now being used to discover novel potent antibiotics (Stokes
et al., 2020), serve estimated travel times in Google Maps (Derrow-Pinion et al.,
2021), power content recommendations in Pinterest (Ying et al., 2018) and product
recommendations in Amazon (Hao et al., 2020), and design the latest generation of
machine learning hardware: the TPUv5 (Mirhoseini et al., 2021). Further, GNN-
based systems have helped mathematicians uncover the hidden structure of mathe-
matical objects (Davies et al., 2021), leading to new top-tier conjectures in the area
of representation theory (Blundell et al., 2021). It would not be an understatement
to say that billions of people are coming into contact with predictions of a GNN,
on a day-to-day basis. As such, it is likely a valuable pursuit to study GNNs, even
without aiming to directly contribute to their development.
Beyond this, it is likely that the very cognition processes driving our reasoning and
decision-making are, in some sense, graph-structured. That is, paraphrasing a quote
from Forrester (1971), nobody really imagines in their head all the information known
to them; rather, they imagine only selected concepts, and relationships between them,
and use those to represent the real system. If we subscribe to this interpretation of
cognition, it is quite unlikely that we will be able to build a generally intelligent
system without some component relying on graph representation learning. Note
that this finding does not clash with the fact that many recent skillful ML systems
are based on the Transformer architecture (Vaswani et al., 2017)—as we will uncover
in this review, Transformers are themselves a special case of GNNs.
2
2 The fundamentals: Permutation equivariance
and invariance
In the previous section, we saw why it is a good idea to study data that lives on
graphs. Now we will see how to learn useful functions over graph-structured data.
The exposition largely follows Bronstein et al. (2021).
With graph-structured inputs, we typically assume a graph G = (V, E); that is,
we have a set of edges E ⊆ V × V, which specifies pairs of nodes in V that are
connected.
As we are interested in representation learning over the nodes, we attach to each
node u ∈ V a feature vector, xu ∈ Rk. The main way in which this data is presented
to a machine learning model is in the form of a node feature matrix. That is, a matrix
X ∈ R|V|×k is prepared by stacking these features:
X = [x1, x2, . . . , x|V|
]> (1)
that is, the ith row of X corresponds to xi.
There are many ways to represent E; since our context is one of linear algebra,
we will use the adjacency matrix, A ∈ R|V|×|V|:
auv =
{
1 (u, v) ∈ E
0 (u, v) /∈ E (2)
Note that it is often possible, especially in biochemical inputs, that we want to
attach more information to the edges (such as distance scalars, or even entire feature
vectors). I deliberately do not consider such cases to retain clarity—the conclusions
we make would be the same in those cases.
However, the very act of using the above representations imposes a node ordering,
and is therefore an arbitrary choice which does not align with the nodes and edges
being unordered! Hence, we need to make sure that permuting the nodes and edges
(PAP>, for a permutation matrix P), does not change the outputs. We recover the
following rules a GNN must satisfy:
f (PX, PAP>) = f (X, A) (Invariance) (3)
F(PX, PAP>) = PF(X, A) (Equivariance) (4)
Here we assumed for simplicity that the functions f , F do not change the adjacency
matrix, so we assume they only return graph or node-level outputs.
3
Further, the graph’s edges allow for a locality constraint in these functions. Much
like how a CNN operates over a small neighbourhood of each pixel of an image, a
GNN can operate over a neighbourhood of a node. One standard way to define this
neighbourhood, Nu, is as follows:
Nu = {v | (u, v) ∈ E ∨ (v, u) ∈ E} (5)
Accordingly, we can define the multiset of all neighbourhood features, XNu :
XNu = {{xv | v ∈ Nu}} (6)
And our local function, φ, can take into account the neighbourhood; that is:
hu = φ(xu, XNu ) F(X) = [h1, h2, . . . , h|V|
]> (7)
Through simple linear algebra manipulation, it is possible to show that if φ is per-
mutation invariant in XNu , then F will be permutation equivariant. The remaining
question is, how do we define φ?
3 Graph Neural Networks
Needless to say, defining φ is one of the most active areas of machine learning research
today. Depending on the literature context, it may be referred to as either “diffusion”,
“propagation”, or “message passing”. As claimed by Bronstein et al. (2021), most
of them can be classified into one of three spatial flavours:
hu = φ
(
xu, ⊕
v∈Nu
cvuψ(xv)
)
(Convolutional) (8)
hu = φ
(
xu, ⊕
v∈Nu
a(xu, xv)ψ(xv)
)
(Attentional) (9)
hu = φ
(
xu, ⊕
v∈Nu
ψ(xu, xv)
)
(Message-passing) (10)
where ψ and φ are neural networks—e.g. ψ(x) = ReLU(Wx + b), and ⊕ is any
permutation-invariant aggregator, such as ∑, averaging, or max. The expressive
power of the GNN progressively increases going from Equation 8 to 10, at the cost
of interpretability, scalability, or learning stability. For most tasks, a careful tradeoff
is needed when choosing the right flavour.
4
This review does not attempt to be a comprehensive overview of specific GNN
layers. That being said: representative convolutional GNNs include the Chebyshev
network (Defferrard et al., 2016, ChebyNet), graph convolutional network (Kipf and
Welling, 2017, GCN) and the simplified graph convolution (Wu et al., 2019, SGC);
representative attentional GNNs include the mixture model CNN (Monti et al., 2017,
MoNet), graph attention network (Veliˇckovi´c et al., 2018, GAT) and its recent “v2”
variant (Brody et al., 2022, GATv2); and representative message-passing GNNs in-
clude interaction networks (Battaglia et al., 2016, IN), message passing neural net-
works (Gilmer et al., 2017, MPNN) and graph networks (Battaglia et al., 2018, GN).
Given such a GNN layer, we can learn (m)any interesting tasks over a graph, by
appropriately combining hu. I exemplify the three principal such tasks, grounded in
biological examples:
Node classification. If the aim is to predict targets for each node u ∈ V, then
our output is equivariant, and we can learn a shared classifier directly on hu. A
canonical example of this is classifying protein functions (e.g. using gene ontology
data (Zitnik and Leskovec, 2017)) in a given protein-protein interaction network, as
first done by GraphSAGE (Hamilton et al., 2017).
Graph classification. If we want to predict targets for the entire graph, then
we want an invariant output, hence need to first reduce all the hu into a common rep-
resentation, e.g. by performing ⊕
u∈V hu, then learning a classifier over the resulting
flat vector. A canonical example is classifying molecules for their quantum-chemical
properties (Gilmer et al., 2017), estimating pharmacological properties like toxicity
or solubility (Duvenaud et al., 2015; Xiong et al., 2019; Jiang et al., 2021) or virtual
drug screening (Stokes et al., 2020).
Link prediction. Lastly, we may be interested in predicting properties of edges
(u, v), or even predicting whether an edge exists; giving rise to the name “link pre-
diction”. In this case, a classifier can be learnt over the concatenation of features
hu‖hv, along with any given edge-level features. Canonical tasks include predict-
ing links between drugs and diseases—drug repurposing (Morselli Gysi et al., 2021),
drugs and targets—binding affinity prediction (Lim et al., 2019; Jiang et al., 2020),
or drugs and drugs—predicting adverse side-effects from polypharmacy (Zitnik et al.,
2018; Deac et al., 2019).
It is possible to use the building blocks from the principal tasks above to go
beyond classifying the entities given by the input graph, and have systems that
produce novel molecules (Mercado et al., 2021) or even perform retrosynthesis—the
estimation of which reactions to utilise to synthesise given molecules (Somnath et al.,
2021; Liu et al., 2022).
A natural question arises, following similar discussions over sets (Zaheer et al.,
5
2017; Wagstaff et al., 2019): Do GNNs, as given by Equation 10, represent all of
the valid permutation-equivariant functions over graphs? Opinions are divided. Key
results in previous years seem to indicate that such models are fundamentally limited
in terms of problems they can solve (Xu et al., 2019; Morris et al., 2019). However,
most, if not all, of the proposals for addressing those limitations are still expressible
using the pairwise message passing formalism of Equation 10; the main requirement
is to carefully modify the graph over which the equation is applied (Veliˇckovi´c, 2022).
To supplement this further, Loukas (2020) showed that, under proper initial features,
sufficient depth-width product (#layers × dim hu), and correct choices of ψ and
φ, GNNs in Equation 10 are Turing universal —likely to be able to simulate any
computation which any computer can perform over such inputs.
All points considered, it is the author’s opinion that the formalism in this section
is likely all we need to build powerful GNNs—although, of course, different perspec-
tives may benefit different problems, and existence of a powerful GNN does not mean
it is easy to find using stochastic gradient descent.
4 GNNs without a graph: Deep Sets and Trans-
formers
Throughout the prior section, we have made a seemingly innocent assumption: that
we are given an input graph (through A). However, very often, not only will there
not be a clear choice of A, but we may not have any prior belief on what A even
is. Further, even if a ground-truth A is given without noise, it may not be the
optimal computation graph: that is, passing messages over it may be problematic—
for example, due to bottlenecks (Alon and Yahav, 2021). As such, it is generally
a useful pursuit to study GNNs that are capable of modulating the input graph
structure.
Accordingly, let us assume we only have a node feature matrix X, but no adja-
cency. One simple option is the “pessimistic” one: assume there are no edges at all,
i.e. A = I, or Nu = {u}. Under such an assumption, Equations 8–10 all reduce to
hu = φ(xu), yielding the Deep Sets model (Zaheer et al., 2017). Therefore, no power
from graph-based modelling is exploited here.
The converse option (the “lazy” one) is to, instead, assume a fully-connected
graph; that is A = 11>, or Nu = V. This then gives the GNN the full potential to
exploit any edges deemed suitable, and is a very popular choice for smaller numbers
of nodes. It can be shown that convolutional GNNs (Equation 8) would still reduce
to Deep Sets in this case, which motivates the use of a stronger GNN. The next model
6
in the hierarchy, attentional GNNs (Equation 9), reduce to the following equation:
hu = φ
(
xu, ⊕
v∈V
a(xu, xv)ψ(xv)
)
(11)
which is essentially the forward pass of a Transformer (Vaswani et al., 2017). To
reverse-engineer why Transformers appear here, let us consider the NLP perspective.
Namely, words in a sentence interact (e.g. subject-object, adverb-verb). Further,
these interactions are not trivial, and certainly not sequential —that is, words can
interact even if they are many sentences apart1. Hence, we may want to use a graph
between them. But what is this graph? Not even annotators tend to agree, and the
optimal graph may well be task-dependant. In such a setting, a common assumption
is to use a complete graph, and let the network infer relations by itself—at this point,
the Transformer is all but rederived. For an in-depth rederivation, see Joshi (2020).
Another reason why Transformers have become such a dominant GNN variant
is the fact that using a fully connected graph structure allows to express all model
computations using dense matrix products, and hence their computations align very
well with current prevalent accelerators (GPUs and TPUs). Further, they have
a more favourable storage complexity than the message passing variant (Equation
10). Accordingly, Transformers can be seen as GNNs that are currently winning the
hardware lottery (Hooker, 2021)!
Before closing this section, it is worth noting a third option to learning a GNN
without an input graph: to infer a graph structure to be used as edges for a GNN.
This is an emerging area known as latent graph inference. It is typically quite chal-
lenging, since edge selection is a non-differentiable operation, and various paradigms
have been proposed in recent years to overcome this challenge: nonparametric (Wang
et al., 2019; Deac et al., 2022), supervised (Veliˇckovi´c et al., 2020), variational (Kipf
et al., 2018), reinforcement (Kazi et al., 2022) and self-supervised learning (Fatemi
et al., 2021).
"""